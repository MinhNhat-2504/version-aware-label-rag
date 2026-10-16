"""Summarize provisional version-pilot reviews with response provenance."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["bm25_chunked", "evidence_set_chunked"], required=True)
    parser.add_argument("--review", type=Path, default=Path("data/curated/version_answer_review_pilot.jsonl"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = {row["qid"]: row for row in read_jsonl(args.review)
              if row["method"] == args.method}
    prompts = {row["qid"]: row for row in read_jsonl(args.prompts)}
    responses = {row["qid"]: row for row in read_jsonl(args.responses)}
    details = {row["qid"]: row for row in json.loads(
        args.scores.read_text(encoding="utf-8"))["details"]}
    if not (set(review) == set(prompts) == set(responses) == set(details)):
        raise ValueError("Review, prompt, response and score QIDs must match")
    rows = []
    for qid, annotation in review.items():
        detail = details[qid]
        if (annotation["status"] != "assistant_provisional_nonexpert" or
                annotation["correct_abstain_decision"] != detail.get("correct_refusal_decision")):
            raise ValueError(f"Review/score mismatch for {qid}")
        rows.append({
            "qid": qid, "task": prompts[qid]["task"],
            "spl_version": prompts[qid]["spl_version"],
            "xml_sha256": prompts[qid]["xml_sha256"],
            "answer_complete": annotation["answer_complete"],
            "correct_abstain_decision": annotation["correct_abstain_decision"],
            "all_cited_claims_supported": annotation["all_cited_claims_supported"],
            "retrieval_complete_evidence": detail.get("retrieval_complete_evidence"),
            "note": annotation["note"],
        })
    report = {
        "method": args.method,
        "status": "assistant provisional nonexpert; not a paper-level outcome",
        "n": len(rows),
        "answer_complete": sum(row["answer_complete"] for row in rows),
        "correct_abstain_decisions": sum(row["correct_abstain_decision"] for row in rows),
        "answerable_retrieval_complete": sum(row["retrieval_complete_evidence"] is True for row in rows),
        "review_sha256": sha256(args.review.read_bytes()).hexdigest(),
        "prompts_sha256": sha256(args.prompts.read_bytes()).hexdigest(),
        "responses_sha256": sha256(args.responses.read_bytes()).hexdigest(),
        "details": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("method", "n", "answer_complete",
          "correct_abstain_decisions", "answerable_retrieval_complete")}))


if __name__ == "__main__":
    main()
