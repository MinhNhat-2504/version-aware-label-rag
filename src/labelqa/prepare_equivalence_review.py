"""Export development-set passage-equivalence decisions for two blind readers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


SOURCE = Path("results/equivalence_review_validation.jsonl")


def main() -> None:
    rows = []
    for item in read_jsonl(SOURCE):
        for candidate in item["near_candidates_unverified"]:
            rows.append({
                "review_id": f"{item['qid']}:{candidate['index']}",
                "qid": item["qid"], "set_id": item["set_id"],
                "question": item["question"],
                "original_index": item["gold_index"],
                "original_passage": item["gold_text"],
                "candidate_index": candidate["index"],
                "candidate_passage": candidate["text"],
                "reviewer_id": "", "decision": "",  # equivalent, not_equivalent, uncertain
                "supported_atomic_fact": "", "rationale": "",
            })
    output_dir = Path("data/review")
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = [output_dir / f"equivalence_validation_reader_{reader}_blank.jsonl"
                    for reader in ("a", "b")]
    destinations.append(output_dir / "equivalence_validation_manifest.json")
    if any(path.exists() for path in destinations):
        raise ValueError("Frozen equivalence review packet exists; refusing to overwrite it")
    for reader in ("a", "b"):
        path = output_dir / f"equivalence_validation_reader_{reader}_blank.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "status": "development_validation_candidates_pending_two_independent_readers",
        "candidate_count": len(rows), "question_count": len({r["qid"] for r in rows}),
        "source": str(SOURCE), "source_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
        "review_instruction": "Judge whether candidate supports the same question-specific atomic fact as original; do not infer equivalence from overlapping words or numbers. Record uncertainty and preserve separate reader files. This is benchmark validation material, not an untouched final test.",
    }
    (output_dir / "equivalence_validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_pairs": len(rows), "questions": manifest["question_count"]}))


if __name__ == "__main__":
    main()
