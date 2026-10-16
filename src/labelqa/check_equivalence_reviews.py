"""Validate two independent passage-equivalence reviews and queue reconciliation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


DECISIONS = {"equivalent", "not_equivalent", "uncertain"}
IDENTITY = ("review_id", "qid", "set_id", "question", "original_index",
            "original_passage", "candidate_index", "candidate_passage")
ANNOTATIONS = ("reviewer_id", "decision", "supported_atomic_fact", "rationale")
DEFAULT_MANIFEST = Path("data/review/equivalence_validation_manifest.json")
DEFAULT_BLANK_A = Path("data/review/equivalence_validation_reader_a_blank.jsonl")
DEFAULT_BLANK_B = Path("data/review/equivalence_validation_reader_b_blank.jsonl")


def indexed(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    data = {row["review_id"]: row for row in rows}
    if len(rows) != len(data):
        raise ValueError(f"Duplicate review_id in {path}")
    return data


def frozen_candidates(manifest_path: Path, blank_a: Path, blank_b: Path) -> dict[str, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "development_validation_candidates_pending_two_independent_readers":
        raise ValueError("Unrecognized frozen equivalence manifest")
    source = Path(manifest["source"])
    if sha256(source.read_bytes()).hexdigest() != manifest.get("source_sha256"):
        raise ValueError("Frozen equivalence source changed")
    if blank_a.read_bytes() != blank_b.read_bytes():
        raise ValueError("Frozen reader blank packets differ")
    rows = indexed(blank_a)
    expected = {}
    for item in read_jsonl(source):
        for candidate in item["near_candidates_unverified"]:
            rid = f"{item['qid']}:{candidate['index']}"
            expected[rid] = {"review_id": rid, "qid": item["qid"],
                             "set_id": item["set_id"], "question": item["question"],
                             "original_index": item["gold_index"],
                             "original_passage": item["gold_text"],
                             "candidate_index": candidate["index"],
                             "candidate_passage": candidate["text"],
                             "reviewer_id": "", "decision": "",
                             "supported_atomic_fact": "", "rationale": ""}
    if (not rows or rows != expected or len(rows) != manifest.get("candidate_count")
            or len({row["qid"] for row in rows.values()}) != manifest.get("question_count")):
        raise ValueError("Blank packet differs from pinned equivalence candidates")
    return rows


def check(blank: dict[str, dict], a: dict[str, dict], b: dict[str, dict]) -> dict:
    if not blank or set(a) != set(blank) or set(b) != set(blank):
        raise ValueError("Both readers must review the frozen candidate set")
    ids_a = {r.get("reviewer_id", "").strip() for r in a.values()
             if isinstance(r.get("reviewer_id", ""), str)}
    ids_b = {r.get("reviewer_id", "").strip() for r in b.values()
             if isinstance(r.get("reviewer_id", ""), str)}
    if any(not isinstance(r.get("reviewer_id", ""), str) for r in (*a.values(), *b.values())):
        raise ValueError("Reviewer IDs must be text")
    if len(ids_a) != 1 or len(ids_b) != 1 or "" in ids_a | ids_b or ids_a == ids_b:
        raise ValueError("Two distinct, nonempty reviewer IDs are required")
    queue = []
    for rid, candidate in blank.items():
        left, right = a[rid], b[rid]
        for review in (left, right):
            if set(review) != set(candidate) or any(review[key] != candidate[key] for key in IDENTITY):
                raise ValueError(f"Frozen candidate identity or fields changed: {rid}")
            if review.get("decision") not in DECISIONS:
                raise ValueError(f"Missing/invalid decision for {rid}")
            if (not isinstance(review.get("supported_atomic_fact"), str)
                    or not review["supported_atomic_fact"].strip()
                    or not isinstance(review.get("rationale"), str)
                    or not review["rationale"].strip()):
                raise ValueError(f"Missing atomic fact/rationale for {rid}")
        queue.append({"review_id": rid, "qid": candidate["qid"],
                      "set_id": candidate["set_id"],
                      "original_index": candidate["original_index"],
                      "candidate_index": candidate["candidate_index"],
                      "reader_a": {k: left[k] for k in ANNOTATIONS},
                      "reader_b": {k: right[k] for k in ANNOTATIONS},
                      "decision_disagreement": left["decision"] != right["decision"],
                      "resolution_mode": "", "consensus_reviewer_ids": [],
                      "expert_resolver_id": "", "adjudicator_id": "",
                      "final_decision": "", "adjudication_notes": ""})
    pairs = {f"{x}/{y}": sum(row["reader_a"]["decision"] == x and
                             row["reader_b"]["decision"] == y for row in queue)
             for x in sorted(DECISIONS) for y in sorted(DECISIONS)}
    n = len(queue)
    exact_count = sum(pairs[f"{decision}/{decision}"] for decision in DECISIONS)
    observed = exact_count / n
    expected = sum((sum(pairs[f"{decision}/{other}"] for other in DECISIONS) / n) *
                   (sum(pairs[f"{other}/{decision}"] for other in DECISIONS) / n)
                   for decision in DECISIONS)
    return {"status": "frozen_equivalence_reviews_checked_manual_adjudication_required",
            "items": n, "reader_ids": [next(iter(ids_a)), next(iter(ids_b))],
            "decision_pairs": pairs,
            "decision_agreement": {"exact_count": exact_count,
                                   "observed_fraction": observed,
                                   "cohen_kappa": (observed - expected) / (1 - expected)
                                   if expected < 1 else None,
                                   "unit": "candidate passage pair; pairs sharing a question are dependent"},
            "all_items_require_manual_adjudication": True, "queue": queue}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-a", type=Path, required=True)
    parser.add_argument("--reader-b", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--blank-a", type=Path, default=DEFAULT_BLANK_A)
    parser.add_argument("--blank-b", type=Path, default=DEFAULT_BLANK_B)
    parser.add_argument("--output", type=Path, default=Path("data/review/equivalence_adjudication_queue.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an equivalence review queue")
    blank = frozen_candidates(args.manifest, args.blank_a, args.blank_b)
    report = check(blank, indexed(args.reader_a), indexed(args.reader_b))
    report["source_sha256"] = {str(path): sha256(path.read_bytes()).hexdigest() for path in
                               (args.manifest, args.blank_a, args.blank_b,
                                args.reader_a, args.reader_b)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": report["items"], "disagreements": sum(r["decision_disagreement"] for r in report["queue"]),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
