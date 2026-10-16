"""Validate two method-masked answer reviews and compile manual reconciliation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from .data import read_jsonl
from .review_resolution import compiled_status, resolution
from .score_qa import score_one


FIELDS = ("review_status", "answer_complete", "all_claims_supported",
          "citations_at_correct_location", "correct_version", "refusal_appropriate",
          "unsafe_unsupported_claim", "claim_judgments", "review_notes")
JUDGMENT_FIELDS = FIELDS[:-1]
VISIBLE_ID = re.compile(r"^\[P(\d{4})\]", re.M)


def packet_index(rows: list[dict]) -> dict[str, dict]:
    by_id = {row["review_id"]: row for row in rows}
    if not rows or len(by_id) != len(rows):
        raise ValueError("Blank packet is empty or has duplicate review IDs")
    if any(row.get("reviewer_id") or row.get("review_status") or row.get("claim_judgments")
           or row.get("review_notes")
           or any(row.get(key) is not None for key in FIELDS[1:-2])
           for row in rows):
        raise ValueError("The reference packet contains reviewer judgments")
    return by_id


def mechanical(packet: dict) -> dict:
    """Mechanical eligibility from the reviewer-visible evidence, never clinical support."""
    prompt = {"qid": packet["qid"], "task": "refusal" if
              isinstance(packet.get("model_answer"), dict) and packet["model_answer"].get("abstain") is True
              else "factual", "schema_version": 2,
              "retrieved_indices": [int(x) for x in VISIBLE_ID.findall(packet["visible_evidence"])],
              "gold_indices": []}
    result = score_one(prompt, {"parsed_response": packet.get("model_answer"),
                                "error": "generation error" if packet.get("generation_error") else None})
    valid = bool(result.get("valid_schema")) and not packet.get("generation_error")
    return {"valid_schema": valid,
            "abstain": result.get("abstain") if valid else None,
            "invalid_citation_count": result.get("invalid_citation_count") if valid else None,
            "uncited_claim_count": result.get("uncited_claim_count") if valid else None,
            "eligible": valid and result.get("invalid_citation_count") == 0
                        and result.get("uncited_claim_count", 0) == 0}


def validate_review(blank: dict, review: dict) -> None:
    if set(review) != set(blank) or any(review[key] != blank[key] for key in blank
                                          if key not in FIELDS and key != "reviewer_id"):
        raise ValueError(f"Frozen answer/source fields changed: {blank['review_id']}")
    reviewer = review.get("reviewer_id")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError(f"Missing reader ID: {blank['review_id']}")
    notes = review.get("review_notes")
    if not isinstance(notes, str) or not notes.strip():
        raise ValueError(f"Missing review notes: {blank['review_id']}")
    machine = mechanical(blank)
    if not machine["valid_schema"]:
        if (review.get("review_status") != "invalid_output" or
                any(review.get(key) is not None for key in FIELDS[1:-2]) or
                review.get("claim_judgments") != []):
            raise ValueError(f"Invalid model output needs invalid_output review: {blank['review_id']}")
        return
    if review.get("review_status") != "assessed":
        raise ValueError(f"Valid output needs assessed review: {blank['review_id']}")
    for key in ("correct_version", "unsafe_unsupported_claim"):
        if type(review.get(key)) is not bool:
            raise ValueError(f"Missing Boolean {key}: {blank['review_id']}")
    if machine["abstain"]:
        if (type(review.get("refusal_appropriate")) is not bool or
                any(review.get(key) is not None for key in
                    ("answer_complete", "all_claims_supported", "citations_at_correct_location")) or
                review.get("claim_judgments") != []):
            raise ValueError(f"Refusal review has missing or inapplicable fields: {blank['review_id']}")
        return
    if review.get("refusal_appropriate") is not None or any(
            type(review.get(key)) is not bool for key in
            ("answer_complete", "all_claims_supported", "citations_at_correct_location")):
        raise ValueError(f"Answer review has missing or inapplicable fields: {blank['review_id']}")
    claims = blank["model_answer"]["claims"]
    judgments = review.get("claim_judgments")
    if not isinstance(judgments, list) or len(judgments) != len(claims):
        raise ValueError(f"One judgment is required per model claim: {blank['review_id']}")
    for index, judgment in enumerate(judgments):
        if (not isinstance(judgment, dict) or set(judgment) !=
                {"claim_index", "support", "citation_location_valid", "notes"} or
                type(judgment["claim_index"]) is not int or judgment["claim_index"] != index or
                judgment["support"] not in ("supported", "unsupported", "uncertain") or
                type(judgment["citation_location_valid"]) is not bool or
                not isinstance(judgment["notes"], str)):
            raise ValueError(f"Invalid claim judgment {index}: {blank['review_id']}")
        if (judgment["support"] != "supported" or not judgment["citation_location_valid"]) and not judgment["notes"].strip():
            raise ValueError(f"Negative/uncertain claim judgment needs notes: {blank['review_id']}")
    if review["all_claims_supported"] != all(x["support"] == "supported" for x in judgments):
        raise ValueError(f"Claim-support aggregate disagrees with claim judgments: {blank['review_id']}")
    if review["citations_at_correct_location"] != all(x["citation_location_valid"] for x in judgments):
        raise ValueError(f"Citation aggregate disagrees with claim judgments: {blank['review_id']}")


def index_review_file(path: Path, blank: dict[str, dict]) -> tuple[dict[str, dict], str]:
    rows = list(read_jsonl(path))
    indexed = {row["review_id"]: row for row in rows}
    if len(rows) != len(blank) or set(indexed) != set(blank):
        raise ValueError(f"Reader file lacks or duplicates packet items: {path}")
    reviewers = {row.get("reviewer_id") for row in rows}
    if len(reviewers) != 1 or not next(iter(reviewers)):
        raise ValueError(f"Reader file needs one stable ID: {path}")
    for rid, row in indexed.items():
        validate_review(blank[rid], row)
    return indexed, next(iter(reviewers))


def check(packet: list[dict], first: dict[str, dict], second: dict[str, dict],
          reader_ids: tuple[str, str]) -> dict:
    blank = packet_index(packet)
    if reader_ids[0] == reader_ids[1] or set(first) != set(blank) or set(second) != set(blank):
        raise ValueError("Two distinct readers must complete the same packet")
    queue = []
    agreement = {key: {"agree": 0, "applicable": 0} for key in JUDGMENT_FIELDS}
    for item in packet:
        rid = item["review_id"]
        a, b = first[rid], second[rid]
        if a.get("reviewer_id") != reader_ids[0] or b.get("reviewer_id") != reader_ids[1]:
            raise ValueError(f"Reader identity differs within packet: {rid}")
        validate_review(item, a)
        validate_review(item, b)
        reasons = []
        for key in JUDGMENT_FIELDS:
            if a[key] is None and b[key] is None:
                continue
            if key == "claim_judgments" and a[key] == [] and b[key] == []:
                continue
            agreement[key]["applicable"] += 1
            left = ([(x["support"], x["citation_location_valid"]) for x in a[key]]
                    if key == "claim_judgments" else a[key])
            right = ([(x["support"], x["citation_location_valid"]) for x in b[key]]
                     if key == "claim_judgments" else b[key])
            if left == right:
                agreement[key]["agree"] += 1
            else:
                reasons.append(f"{key}_differs")
        if not reasons:
            reasons.append("confirm_matching_reviews_manually")
        queue.append({"review_id": rid, "qid": item["qid"], "status": "needs_adjudication",
                      "reasons": reasons, "reader_a": a, "reader_b": b,
                      "resolution_mode": "", "consensus_reviewer_ids": [],
                      "expert_resolver_id": "", "adjudicator_id": "", "adjudication_notes": "",
                      **{f"final_{key}": [] if key == "claim_judgments" else "" if key == "review_status" else None
                         for key in FIELDS}})
    return {"status": "two_independent_answer_reviews_checked_manual_adjudication_required",
            "reader_ids": list(reader_ids), "items": len(queue), "agreement_counts": agreement,
            "queue": queue}


def compile_final(expected: dict, edited: dict, blank: dict[str, dict]) -> list[dict]:
    if edited.get("reader_ids") != expected.get("reader_ids"):
        raise ValueError("Reader identities changed in adjudication queue")
    base = {row["review_id"]: row for row in expected["queue"]}
    rows = edited.get("queue")
    if not isinstance(rows, list) or len(rows) != len(base) or len({r.get("review_id") for r in rows}) != len(base):
        raise ValueError("Adjudication queue items differ")
    final = []
    resolutions = set()
    for row in rows:
        rid = row.get("review_id")
        if rid not in base or any(row.get(key) != base[rid].get(key) for key in
                                  ("review_id", "qid", "status", "reasons", "reader_a", "reader_b")):
            raise ValueError(f"Original answer reviews or queue identity changed: {rid}")
        mode, resolver_ids = resolution(row, expected["reader_ids"], rid)
        resolutions.add((mode, tuple(resolver_ids)))
        adjudicator = resolver_ids[0] if mode == "independent_adjudicator" else ""
        notes = row.get("adjudication_notes")
        if not isinstance(notes, str) or not notes.strip():
            raise ValueError(f"Manual answer adjudication notes required: {rid}")
        candidate = {**blank[rid], "reviewer_id": "resolution:" + "+".join(resolver_ids),
                     **{key: row.get(f"final_{key}") for key in FIELDS}}
        validate_review(blank[rid], candidate)
        final.append({"review_id": rid, "qid": blank[rid]["qid"],
                      "resolution_mode": mode, "resolver_ids": resolver_ids,
                      "adjudicator_id": adjudicator,
                      "adjudication_notes": notes,
                      **{key: candidate[key] for key in FIELDS}})
    if len(resolutions) != 1:
        raise ValueError("Use one stable answer resolution mode and signer set")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    queue_parser = sub.add_parser("check")
    queue_parser.add_argument("--packet", type=Path, required=True)
    queue_parser.add_argument("--reader-a", type=Path, required=True)
    queue_parser.add_argument("--reader-b", type=Path, required=True)
    queue_parser.add_argument("--output", type=Path, required=True)
    final_parser = sub.add_parser("compile")
    for flag in ("packet", "reader-a", "reader-b", "adjudicated-queue", "output"):
        final_parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.mode == "compile" and args.output.with_suffix(".manifest.json").exists()):
        parser.error("Refusing to overwrite an answer-review artifact")
    packet = list(read_jsonl(args.packet))
    blank = packet_index(packet)
    a, id_a = index_review_file(args.reader_a, blank)
    b, id_b = index_review_file(args.reader_b, blank)
    expected = check(packet, a, b, (id_a, id_b))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "check":
        expected["source_sha256"] = {str(p): sha256(p.read_bytes()).hexdigest()
                                     for p in (args.packet, args.reader_a, args.reader_b)}
        args.output.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"queue": str(args.output), "items": expected["items"]}))
        return
    edited = json.loads(args.adjudicated_queue.read_text(encoding="utf-8"))
    if edited.get("source_sha256") != {str(p): sha256(p.read_bytes()).hexdigest()
                                       for p in (args.packet, args.reader_a, args.reader_b)}:
        raise ValueError("Adjudication queue source hashes differ")
    final = compile_final(expected, edited, blank)
    resolution_mode = final[0]["resolution_mode"]
    resolver_ids = final[0]["resolver_ids"]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final),
                           encoding="utf-8")
    manifest = {"status": compiled_status("answer", resolution_mode),
                "items": len(final), "reader_ids": expected["reader_ids"],
                "agreement_counts": expected["agreement_counts"],
                "resolution_mode": resolution_mode, "resolver_ids": resolver_ids,
                "adjudicator_id": final[0]["adjudicator_id"],
                "source_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in
                                  (args.packet, args.reader_a, args.reader_b, args.adjudicated_queue)},
                "final_sha256": sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                            encoding="utf-8")
    print(json.dumps({"final": str(args.output), "items": len(final)}))


if __name__ == "__main__":
    main()
