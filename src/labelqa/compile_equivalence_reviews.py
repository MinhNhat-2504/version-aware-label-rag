"""Compile every manually adjudicated equivalence decision with source receipts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .check_equivalence_reviews import (DECISIONS, DEFAULT_BLANK_A, DEFAULT_BLANK_B,
                                        DEFAULT_MANIFEST, check, frozen_candidates, indexed)
from .review_resolution import resolution


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def compile_final(expected: dict, edited: dict, blank: dict[str, dict]) -> list[dict]:
    if (edited.get("status") != expected["status"] or
            edited.get("reader_ids") != expected["reader_ids"] or
            edited.get("items") != expected["items"] or
            edited.get("decision_pairs") != expected["decision_pairs"] or
            edited.get("decision_agreement") != expected["decision_agreement"] or
            edited.get("all_items_require_manual_adjudication") is not True):
        raise ValueError("Frozen equivalence queue metadata changed")
    original = {row["review_id"]: row for row in expected["queue"]}
    rows = edited.get("queue")
    if (not isinstance(rows, list) or len(rows) != len(original) or
            len({row.get("review_id") for row in rows}) != len(original)):
        raise ValueError("Equivalence adjudication items differ")
    final = []
    resolutions = set()
    for row in rows:
        rid = row.get("review_id")
        if rid not in original or any(row.get(key) != original[rid][key] for key in
                                      ("review_id", "qid", "set_id", "original_index",
                                       "candidate_index", "reader_a", "reader_b",
                                       "decision_disagreement")):
            raise ValueError(f"Original equivalence review changed: {rid}")
        mode, resolver_ids = resolution(row, expected["reader_ids"], rid)
        resolutions.add((mode, tuple(resolver_ids)))
        adjudicator = resolver_ids[0] if mode == "independent_adjudicator" else ""
        notes = row.get("adjudication_notes")
        decision = row.get("final_decision")
        if not isinstance(notes, str) or not notes.strip() or decision not in DECISIONS:
            raise ValueError(f"Manual equivalence resolution required: {rid}")
        source = blank[rid]
        final.append({"review_id": rid, "qid": source["qid"], "set_id": source["set_id"],
                      "gold_index": source["original_index"],
                      "candidate_index": source["candidate_index"],
                      "decision": {"equivalent": "accept", "not_equivalent": "reject",
                                   "uncertain": "uncertain"}[decision],
                      "resolution_mode": mode, "resolver_ids": resolver_ids,
                      "adjudicator_id": adjudicator, "adjudication_notes": notes})
    if len(resolutions) != 1:
        raise ValueError("Use one stable resolution mode and signer set")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--blank-a", type=Path, default=DEFAULT_BLANK_A)
    parser.add_argument("--blank-b", type=Path, default=DEFAULT_BLANK_B)
    parser.add_argument("--reader-a", type=Path, required=True)
    parser.add_argument("--reader-b", type=Path, required=True)
    parser.add_argument("--adjudicated-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        parser.error("Refusing to overwrite compiled equivalence reviews")
    paths = (args.manifest, args.blank_a, args.blank_b, args.reader_a, args.reader_b)
    blank = frozen_candidates(args.manifest, args.blank_a, args.blank_b)
    expected = check(blank, indexed(args.reader_a), indexed(args.reader_b))
    edited = json.loads(args.adjudicated_queue.read_text(encoding="utf-8"))
    source_hashes = {str(path): digest(path) for path in paths}
    if edited.get("source_sha256") != source_hashes:
        raise ValueError("Equivalence review queue source receipts changed")
    final = compile_final(expected, edited, blank)
    resolution_mode = final[0]["resolution_mode"]
    resolver_ids = final[0]["resolver_ids"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
                           encoding="utf-8")
    receipt = {"status": ("two_reader_equivalence_consensus_compiled_no_automatic_agreement_to_gold"
                           if resolution_mode == "two_reader_consensus"
                           else "manual_equivalence_adjudication_compiled_no_automatic_agreement_to_gold"),
               "items": len(final), "reader_ids": expected["reader_ids"],
               "resolution_mode": resolution_mode, "resolver_ids": resolver_ids,
               "adjudicator_id": final[0]["adjudicator_id"],
               "frozen_manifest": str(args.manifest),
               "blank_a": str(args.blank_a), "blank_b": str(args.blank_b),
               "decision_counts": {decision: sum(row["decision"] == decision for row in final)
                                   for decision in ("accept", "reject", "uncertain")},
               "reader_decision_pairs": expected["decision_pairs"],
               "reader_decision_agreement": expected["decision_agreement"],
               "source_sha256": {**source_hashes, str(args.adjudicated_queue): digest(args.adjudicated_queue)},
               "final_sha256": digest(args.output)}
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "items": len(final),
                      "decision_counts": receipt["decision_counts"]}))


if __name__ == "__main__":
    main()
