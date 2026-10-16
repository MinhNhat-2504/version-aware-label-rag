"""Verify frozen holdout provenance, disjointness and question/source identity."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .build_version_holdout import PRIOR_SNAPSHOTS
from .data import load_questions, portable_path, read_jsonl


EXPANSION_MANIFEST = Path("data/holdout/version_expansion_manifest.json")
COHORTS = (
    (Path("data/snapshots/version_holdout_sources.jsonl"), Path("data/holdout/version_test_manifest.json")),
    (Path("data/snapshots/version_change_cohort_sources.jsonl"), Path("data/holdout/version_change_cohort_manifest.json")),
    (Path("data/snapshots/version_expansion_consecutive_sources.jsonl"), EXPANSION_MANIFEST),
    (Path("data/snapshots/version_expansion_change_sources.jsonl"), EXPANSION_MANIFEST),
)


def frozen_receipt(manifest: dict, source: Path) -> tuple[str, int]:
    """Return the frozen checksum and pair count for one source file."""
    if "sources" in manifest:  # The expansion manifest records both of its arms.
        receipt = manifest["sources"].get(source.as_posix())
        if receipt is None:
            raise ValueError(f"Source is absent from its frozen manifest: {source}")
        return receipt["sha256"], receipt["selected_pairs"]
    return manifest["sources_sha256"], manifest["selected_pairs"]


def validate_candidate_pairing(candidates: list[dict], source_by_set_id: dict[str, Path]) -> None:
    """Each distinct question must appear once in each of the two version slots."""
    by_pair: dict[str, list[dict]] = {}
    for row in candidates:
        set_id = row["set_id"]
        if (row.get("pair_id") != set_id or set_id not in source_by_set_id
                or portable_path(row["source_snapshot"]) != source_by_set_id[set_id]):
            raise ValueError(f"Candidate pair/source identity mismatch: {row['qid']}")
        by_pair.setdefault(set_id, []).append(row)
    if set(by_pair) != set(source_by_set_id):
        raise ValueError("Candidate pairs do not cover all frozen source pairs")
    for set_id, rows in by_pair.items():
        versions = {row["spl_version"] for row in rows}
        if len(versions) != 2:
            raise ValueError(f"Pair needs two distinct SPL versions: {set_id}")
        by_question: dict[str, list[dict]] = {}
        for row in rows:
            question = row["question"].strip()
            if not question:
                raise ValueError(f"Blank candidate question: {set_id}")
            by_question.setdefault(question, []).append(row)
        if any(len(items) != 2 or {item["spl_version"] for item in items} != versions
               for items in by_question.values()):
            raise ValueError(f"Question wording differs across versions: {set_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path,
                        default=Path("data/holdout/version_question_candidates_expanded.jsonl"))
    parser.add_argument("--reader-a-blank", type=Path,
                        default=Path("data/review/version_holdout_expanded_reader_a_blank.jsonl"))
    parser.add_argument("--reader-b-blank", type=Path,
                        default=Path("data/review/version_holdout_expanded_reader_b_blank.jsonl"))
    parser.add_argument("--original-only", action="store_true",
                        help="Validate the superseded 12-pair design and its frozen 24-item files")
    parser.add_argument("--output", type=Path,
                        help="Optional metadata-only preflight report; refuses to overwrite")
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("Refusing to overwrite a frozen preflight report")
    prior = {q.set_id for q in load_questions(Path("data/upstream/FDARxBench/data/qa/qa.jsonl"))}
    for path in PRIOR_SNAPSHOTS:
        prior.update(row["set_id"] for row in read_jsonl(path))
    if args.original_only:
        args.candidates = Path("data/holdout/version_question_candidates.jsonl")
        args.reader_a_blank = Path("data/review/version_holdout_reader_a_blank.jsonl")
        args.reader_b_blank = Path("data/review/version_holdout_reader_b_blank.jsonl")
    cohorts = COHORTS[:2] if args.original_only else COHORTS
    pairs = []
    source_by_set_id = {}
    for source, manifest_path in cohorts:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha, expected_pairs = frozen_receipt(manifest, source)
        if sha256(source.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"Frozen source checksum mismatch: {source}")
        rows = list(read_jsonl(source))
        if len(rows) != expected_pairs:
            raise ValueError(f"Frozen source count mismatch: {source}")
        pairs.extend(rows)
        source_by_set_id.update({row["set_id"]: source for row in rows})
    ids = [row["set_id"] for row in pairs]
    if len(ids) != len(set(ids)) or set(ids) & prior:
        raise ValueError("Holdout labels overlap each other or previous development data")
    by_version = {}
    for pair in pairs:
        versions = pair["versions"]
        if len(versions) != 2 or versions[1]["spl_version"] != versions[0]["spl_version"] + 1:
            raise ValueError(f"Version pair is not consecutive: {pair['set_id']}")
        for version in versions:
            if len(version["xml_sha256"]) != 64 or not version["sections"]:
                raise ValueError(f"Missing pinned source data: {pair['set_id']}")
            by_version[(pair["set_id"], version["spl_version"])] = version["xml_sha256"]
    candidates = list(read_jsonl(args.candidates))
    if len(candidates) < len(by_version) or len({r["qid"] for r in candidates}) != len(candidates):
        raise ValueError("Candidate questions are incomplete or duplicated")
    for row in candidates:
        if by_version.get((row["set_id"], row["spl_version"])) != row["xml_sha256"]:
            raise ValueError(f"Candidate source mismatch: {row['qid']}")
        if row["annotation_status"] != "candidate_pending_two_independent_readers":
            raise ValueError(f"Candidate adjudication status changed: {row['qid']}")
        if "reference_answer" in row or "claims" in row:
            raise ValueError(f"Candidate contains gold answer fields: {row['qid']}")
    validate_candidate_pairing(candidates, source_by_set_id)
    qids = {row["qid"] for row in candidates}
    review_paths = []
    for path in (args.reader_a_blank, args.reader_b_blank):
        review_paths.append(path)
        reviews = list(read_jsonl(path))
        if len(reviews) != len(candidates) or {row["qid"] for row in reviews} != qids:
            raise ValueError(f"Blank reviewer packet does not match candidates: {path}")
        if any(row.get("reviewer_id") or row.get("item_decision")
               or row.get("acceptable_evidence_spans_per_claim") != [] for row in reviews):
            raise ValueError(f"Reviewer packet has prefilled decisions/evidence: {path}")
    report = {"status": "source_metadata_preflight_only_no_model_or_human_gold",
              "pairs": len(pairs), "versions": len(by_version),
              "candidate_questions": len(candidates),
              "cohorts": {source.as_posix(): len([pair for pair in pairs
                                                  if source_by_set_id[pair["set_id"]] == source])
                          for source, _ in cohorts},
              "paired_question_wording_verified": True,
              "disjoint_from_benchmark_and_prior_pilots": True,
              "source_snapshot_sha256": {source.as_posix(): sha256(source.read_bytes()).hexdigest()
                                         for source, _ in cohorts},
              "candidate_sha256": sha256(args.candidates.read_bytes()).hexdigest(),
              "blank_review_packet_sha256": {str(path): sha256(path.read_bytes()).hexdigest()
                                              for path in review_paths},
              "independent_human_adjudications": 0,
              "holdout_model_evaluations": 0,
              "note": "This verifies source metadata and question pairing, not claim gold or answer quality."}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "pairs": report["pairs"],
                      "versions": report["versions"], "candidate_questions": report["candidate_questions"],
                      "independent_human_adjudications": 0,
                      "output": str(args.output) if args.output else None}))


if __name__ == "__main__":
    main()
