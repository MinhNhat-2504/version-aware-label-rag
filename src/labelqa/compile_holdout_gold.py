"""Compile signed manual adjudications into frozen gold; never infer agreement."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .check_holdout_reviews import CHANGE_JUDGMENTS, check, validate_evidence_spans
from .data import portable_path, read_jsonl
from .review_resolution import resolution


IMMUTABLE_QUEUE_FIELDS = ("qid", "set_id", "spl_version", "status", "reasons",
                          "reader_a", "reader_b")


def compile_gold(expected: dict, edited: dict, candidates: dict[str, dict],
                 source_sections: dict[str, dict[int, str]]) -> tuple[list[dict], list[dict]]:
    """Require a human decision on every item, including matching reader reviews."""
    base = {row["qid"]: row for row in expected["queue"]}
    rows = edited.get("queue")
    if not isinstance(rows, list) or len(rows) != len(base) or len({r.get("qid") for r in rows}) != len(base):
        raise ValueError("Adjudication queue items differ from the checked reader packet")
    accepted, excluded = [], []
    resolutions = set()
    for row in rows:
        qid = row.get("qid")
        if qid not in base or any(row.get(key) != base[qid].get(key) for key in IMMUTABLE_QUEUE_FIELDS):
            raise ValueError(f"Reader evidence or queue identity was edited for {qid}")
        reviewer_ids = [base[qid][side]["reviewer_id"] for side in ("reader_a", "reader_b")]
        mode, resolver_ids = resolution(row, reviewer_ids, qid)
        resolutions.add((mode, tuple(resolver_ids)))
        adjudicator = resolver_ids[0] if mode == "independent_adjudicator" else ""
        decision = row.get("final_decision")
        if decision == "rewrite":
            raise ValueError(f"Rewrite {qid} in a new frozen candidate file and re-review both readers")
        if decision not in ("accept", "reject"):
            raise ValueError(f"A manual accept/reject decision is required for {qid}")
        notes = row.get("adjudication_notes", "").strip()
        if not notes:
            raise ValueError(f"Manual adjudication notes are required for {qid}")
        candidate = candidates[qid]
        if decision == "reject":
            excluded.append({"qid": qid, "set_id": candidate["set_id"],
                             "spl_version": candidate["spl_version"],
                             "source_snapshot": candidate["source_snapshot"],
                             "reason": notes, "resolution_mode": mode,
                             "resolver_ids": resolver_ids, "adjudicator_id": adjudicator})
            continue
        answerable = row.get("final_answerable_from_version")
        answer = row.get("final_answer", "")
        claims = row.get("final_claims")
        evidence = row.get("final_acceptable_section_indices_per_claim")
        spans = row.get("final_acceptable_evidence_spans_per_claim")
        change = row.get("final_version_change_judgment")
        if type(answerable) is not bool or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Final answer/answerability is incomplete for {qid}")
        if change not in CHANGE_JUDGMENTS:
            raise ValueError(f"Final version-change judgment is missing for {qid}")
        if not isinstance(claims, list) or not isinstance(evidence, list) or not isinstance(spans, list):
            raise ValueError(f"Final claims/evidence arrays are missing for {qid}")
        if answerable:
            if not claims or len(claims) != len(evidence):
                raise ValueError(f"Final claim/evidence alignment is incomplete for {qid}")
            if any(not isinstance(claim, str) or not claim.strip() for claim in claims):
                raise ValueError(f"Empty final claim for {qid}")
            for group in evidence:
                if (not isinstance(group, list) or not group
                        or any(type(index) is not int or index not in source_sections[qid]
                               for index in group)):
                    raise ValueError(f"Final evidence references an absent section for {qid}")
            validate_evidence_spans(qid, claims, evidence, spans, source_sections[qid])
        elif claims or evidence or spans:
            raise ValueError(f"Unanswerable item has positive final claims for {qid}")
        accepted.append({
            "qid": qid, "pair_id": candidate["pair_id"], "set_id": candidate["set_id"],
            "spl_version": candidate["spl_version"], "xml_sha256": candidate["xml_sha256"],
            "source_snapshot": candidate["source_snapshot"], "question": candidate["question"],
            "answerable": answerable, "reference_answer": answer.strip(),
            "claims": [{"text": claim.strip(), "acceptable_section_indices": group,
                        "acceptable_evidence_spans": span_group}
                       for claim, group, span_group in zip(claims, evidence, spans)],
            "version_change_judgment": change, "adjudicator_id": adjudicator,
            "resolution_mode": mode, "resolver_ids": resolver_ids,
            "adjudication_notes": notes,
        })
    if len(resolutions) != 1:
        raise ValueError("Use one stable resolution mode and signer set for all holdout items")
    return accepted, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/holdout/version_question_candidates_expanded.jsonl"))
    parser.add_argument("--reader-a", type=Path, required=True)
    parser.add_argument("--reader-b", type=Path, required=True)
    parser.add_argument("--adjudicated-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/holdout/version_gold_adjudicated.jsonl"))
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Frozen output already exists; choose a new versioned path")
    expected = check(args.candidates, args.reader_a, args.reader_b)
    edited = json.loads(args.adjudicated_queue.read_text(encoding="utf-8"))
    if edited.get("reader_ids") != expected["reader_ids"]:
        raise ValueError("Adjudicated queue reader IDs differ from checked reviews")
    candidates = {row["qid"]: row for row in read_jsonl(args.candidates)}
    source_sections = {}
    source_index = {}
    for source in {portable_path(row["source_snapshot"]) for row in candidates.values()}:
        for pair in read_jsonl(source):
            for version in pair["versions"]:
                source_index[(pair["set_id"], version["spl_version"])] = version
    for qid, candidate in candidates.items():
        version = source_index[(candidate["set_id"], candidate["spl_version"])]
        if version["xml_sha256"] != candidate["xml_sha256"]:
            raise ValueError(f"Pinned source hash changed for {qid}")
        source_sections[qid] = {section["section_index"]: section["text"]
                                for section in version["sections"]}
    accepted, excluded = compile_gold(expected, edited, candidates, source_sections)
    resolved_rows = accepted + excluded
    resolution_mode = resolved_rows[0]["resolution_mode"]
    resolver_ids = resolved_rows[0]["resolver_ids"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": ("two_reader_consensus_compiled_no_automatic_agreement_to_gold"
                   if resolution_mode == "two_reader_consensus"
                   else "manual_adjudication_compiled_no_automatic_agreement_to_gold"),
        "candidate_items": len(candidates),
        "accepted": len(accepted), "excluded": len(excluded), "exclusions": excluded,
        "reader_ids": expected["reader_ids"],
        "resolution_mode": resolution_mode, "resolver_ids": resolver_ids,
        "source_sha256": {str(path): sha256(path.read_bytes()).hexdigest()
                          for path in (args.candidates, args.reader_a, args.reader_b, args.adjudicated_queue)},
        "gold_sha256": sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path),
                      "accepted": len(accepted), "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
