"""Summarize provisional human-readable QA judgments against frozen responses."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=Path("data/curated/temporal_answer_review_pilot.jsonl"))
    parser.add_argument("--method", choices=["bm25", "ontology", "evidence_set"], required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = {row["qid"]: row for row in read_jsonl(args.review)}
    responses = {row["qid"]: row for row in read_jsonl(args.responses)}
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    details = {row["qid"]: row for row in scores["details"]}
    if set(review) != set(responses) or set(review) != set(details):
        raise ValueError("Review, response and score QIDs must match")
    rows = []
    for qid, annotation in review.items():
        if annotation["status"] != "assistant_provisional_nonexpert":
            raise ValueError(f"Unexpected review status for {qid}")
        complete = annotation[f"{args.method}_answer_complete"]
        supported = annotation[f"{args.method}_claims_supported"]
        if not isinstance(complete, bool) or supported not in (True, False, None):
            raise ValueError(f"Invalid review decision for {qid}")
        detail = details[qid]
        rows.append({
            "qid": qid, "answer_complete": complete,
            "all_cited_claims_supported_by_source": supported,
            "schema_valid": detail["valid_schema"],
            "retrieval_complete_evidence": detail["retrieval_complete_evidence"],
            "cited_complete_evidence": detail.get("cited_complete_evidence", False),
            "end_to_end_success": bool(
                complete and supported and detail["valid_schema"]
                and detail.get("cited_complete_evidence")
            ),
            "note": annotation["note"],
        })
    total = len(rows)
    report = {
        "status": "assistant provisional nonexpert; not an independently adjudicated paper metric",
        "method": args.method, "n": total,
        "review_sha256": sha256(args.review.read_bytes()).hexdigest(),
        "responses_sha256": sha256(args.responses.read_bytes()).hexdigest(),
        "scores_sha256": sha256(args.scores.read_bytes()).hexdigest(),
        "answer_complete": sum(row["answer_complete"] for row in rows),
        "all_cited_claims_supported": sum(row["all_cited_claims_supported_by_source"] is True for row in rows),
        "reviewed_claim_response_count": sum(row["all_cited_claims_supported_by_source"] is not None for row in rows),
        "end_to_end_success": sum(row["end_to_end_success"] for row in rows),
        "details": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("method", "n", "answer_complete",
          "all_cited_claims_supported", "reviewed_claim_response_count", "end_to_end_success")}))


if __name__ == "__main__":
    main()
