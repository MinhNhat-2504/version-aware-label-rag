"""Score manually adjudicated answer reviews after the blind method key is opened."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .answer_review import FIELDS, mechanical, packet_index, validate_review
from .data import portable_path, read_jsonl
from .holdout_reporting import COHORTS, cohort_for, sample_flow
from .prepare_holdout_eval import DEFAULT_GOLD, load_manual_gold
from .summarize_holdout_retrieval import METHODS, paired_summary
from .review_resolution import accepted_statuses, review_design, validate_compiled_resolution


def limitations(answer_design: dict) -> list[str]:
    if answer_design["llm_annotator_ids"]:
        support = ("Clinical support is one human expert's signed resolution of their own blind review "
                   "and a disclosed LLM answer review; agreement figures are expert-LLM agreement, "
                   "not inter-human reliability.")
    else:
        support = ("Clinical support is based on preserved independent human reviews and a signed "
                   "manual resolution.")
    return [support, "Small numbers of label pairs make uncertainty intervals imprecise.",
            "The method key is opened only after answer reviews are final."]


def summarize(gold: list[dict], packet: list[dict], private_key: dict,
              final_reviews: list[dict]) -> dict:
    blank = packet_index(packet)
    accepted = {row["qid"]: row for row in gold}
    if not accepted or len(accepted) != len(gold):
        raise ValueError("Manual gold has no accepted items or duplicate question IDs")
    mapping = private_key.get("mapping")
    if not isinstance(mapping, list) or len(mapping) != len(blank) or len({x["review_id"] for x in mapping}) != len(blank):
        raise ValueError("Private method mapping is incomplete or duplicated")
    mapped = {x["review_id"]: x for x in mapping}
    final = {row["review_id"]: row for row in final_reviews}
    if set(mapped) != set(blank) or len(final_reviews) != len(blank) or set(final) != set(blank):
        raise ValueError("Final reviews, blank packet and private mapping differ")
    if len(blank) != len(accepted) * len(METHODS):
        raise ValueError("Packet does not contain each method for every accepted question")
    per_item = []
    for rid, item in blank.items():
        key, review = mapped[rid], final[rid]
        qid, method = item["qid"], key.get("method")
        if (qid not in accepted or method not in METHODS or key.get("qid") != qid or
                review.get("qid") != qid or
                key.get("raw_response_sha256") != sha256(item["raw_model_output"].encode()).hexdigest()):
            raise ValueError(f"Private key or answer identity differs: {rid}")
        resolver_ids = review.get("resolver_ids")
        if resolver_ids is None and isinstance(review.get("adjudicator_id"), str):
            resolver_ids = [review["adjudicator_id"]]
        if not isinstance(resolver_ids, list) or not resolver_ids or not review.get("adjudication_notes", "").strip():
            raise ValueError(f"Missing manual answer resolution: {rid}")
        reconstructed = {**item, "reviewer_id": "resolution:" + "+".join(resolver_ids),
                         **{field: review.get(field) for field in FIELDS}}
        validate_review(item, reconstructed)
        gold_row = accepted[qid]
        if any(item[field] != gold_row[field] for field in
               ("set_id", "spl_version", "xml_sha256", "question", "source_snapshot")):
            raise ValueError(f"Answer packet differs from manual gold: {rid}")
        machine = mechanical(item)
        answerable = gold_row["answerable"]
        if not machine["eligible"]:
            success = False
        elif answerable:
            success = bool(machine["abstain"] is False and review["review_status"] == "assessed"
                           and review["answer_complete"] and review["all_claims_supported"]
                           and review["citations_at_correct_location"] and review["correct_version"]
                           and review["unsafe_unsupported_claim"] is False)
        else:
            success = bool(machine["abstain"] is True and review["review_status"] == "assessed"
                           and review["refusal_appropriate"] and review["correct_version"]
                           and review["unsafe_unsupported_claim"] is False)
        per_item.append({"review_id": rid, "qid": qid, "set_id": gold_row["set_id"],
                         "spl_version": gold_row["spl_version"], "method": method,
                         "cohort": cohort_for(gold_row["source_snapshot"]),
                         "answerable": answerable, "machine": machine,
                         "end_to_end_success": success,
                         "false_refusal": bool(answerable and machine["abstain"] is True),
                         "unsafe_unsupported_claim": review["unsafe_unsupported_claim"],
                         "answer_complete": review["answer_complete"],
                         "all_claims_supported": review["all_claims_supported"],
                         "citations_at_correct_location": review["citations_at_correct_location"],
                         "correct_version": review["correct_version"],
                         "refusal_appropriate": review["refusal_appropriate"],
                         "claim_judgments": review["claim_judgments"]})
    by_method_qid = {}
    for row in per_item:
        key = (row["method"], row["qid"])
        if key in by_method_qid:
            raise ValueError(f"Duplicate method/question in private map: {key}")
        by_method_qid[key] = row
    if set(by_method_qid) != {(method, qid) for method in METHODS for qid in accepted}:
        raise ValueError("Private map lacks a method/question cell")
    def counts_for(subset: list[dict]) -> dict:
        counts = {}
        for method in METHODS:
            rows = [row for row in subset if row["method"] == method]
            generated_claims = [claim for row in rows for claim in row["claim_judgments"]]
            counts[method] = {
            "accepted_items": len(rows),
            "answerable_items": sum(row["answerable"] for row in rows),
            "unanswerable_items": sum(not row["answerable"] for row in rows),
            "end_to_end_successes": sum(row["end_to_end_success"] for row in rows),
            "complete_supported_answerable": sum(row["end_to_end_success"]
                                                 for row in rows if row["answerable"]),
            "appropriate_refusals": sum(row["end_to_end_success"]
                                        for row in rows if not row["answerable"]),
            "false_refusals": sum(row["false_refusal"] for row in rows),
            "invalid_or_unusable_outputs": sum(not row["machine"]["eligible"] for row in rows),
            "unsafe_unsupported_answers": sum(row["unsafe_unsupported_claim"] is True for row in rows),
            "supported_generated_claims": sum(c["support"] == "supported" and c["citation_location_valid"]
                                              for c in generated_claims),
            "reviewed_generated_claims": len(generated_claims),
            }
        return counts
    method_counts = counts_for(per_item)
    def compare(subset: list[str]) -> dict | None:
        if not subset:
            return None
        paired = [{"set_id": gold_row["set_id"],
                   **{method: by_method_qid[(method, qid)]["end_to_end_success"]
                      for method in METHODS}}
                  for qid, gold_row in accepted.items() if qid in subset]
        return {"learned_vs_bm25": paired_summary(paired, "learned_cover", "bm25"),
                "learned_vs_ontology": paired_summary(paired, "learned_cover", "ontology_cover")}
    all_qids = list(accepted)
    answerable_qids = [qid for qid, row in accepted.items() if row["answerable"]]
    by_cohort = {}
    for cohort in COHORTS.values():
        cohort_qids = [qid for qid, row in accepted.items()
                       if cohort_for(row["source_snapshot"]) == cohort]
        cohort_answerable_qids = [qid for qid in cohort_qids if accepted[qid]["answerable"]]
        by_cohort[cohort] = {
            "accepted_items": len(cohort_qids),
            "label_pairs": len({accepted[qid]["set_id"] for qid in cohort_qids}),
            "method_counts": counts_for([row for row in per_item if row["cohort"] == cohort]),
            "paired_all_accepted": compare(cohort_qids),
            "paired_answerable_only": compare(cohort_answerable_qids),
        }
    return {"status": "manual_adjudicated_end_to_end_answer_score_after_gold",
            "accepted_items": len(accepted), "method_counts": method_counts,
            "paired_all_accepted": compare(all_qids),
            "paired_answerable_only": compare(answerable_qids),
            "by_cohort": by_cohort,
            "per_item": per_item}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--execution-lock", type=Path, default=Path("paper/HOLDOUT_EXECUTION_LOCK.json"))
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--final-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an answer score")
    gold, gold_receipt = load_manual_gold(args.gold)  # No answer review opened before gold.
    from .lock_holdout_execution import verify_lock
    _, lock_hash = verify_lock(args.execution_lock, gold, args.gold, gold_receipt)
    packet_bytes = args.packet.read_bytes()
    key = json.loads(args.private_map.read_text(encoding="utf-8"))
    if (key.get("status") != "private_method_key_keep_from_reviewers" or
            key.get("gold_sha256") != gold_receipt["gold_sha256"] or
            key.get("execution_lock_sha256") != lock_hash or
            key.get("packet_sha256") != sha256(packet_bytes).hexdigest()):
        raise ValueError("Private method key does not match manual gold and blank packet")
    manifest_path = args.final_reviews.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") not in accepted_statuses("answer") or
            manifest.get("final_sha256") != sha256(args.final_reviews.read_bytes()).hexdigest() or
            manifest.get("items") != len(gold) * len(METHODS) or
            manifest.get("source_sha256", {}).get(str(args.packet)) != sha256(packet_bytes).hexdigest()):
        raise ValueError("Final human answer reviews have no matching compiler manifest")
    for path, digest in manifest["source_sha256"].items():
        if sha256(portable_path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Original answer-review source changed: {path}")
    for name, receipt in (manifest.get("blind_receipts") or {}).items():
        path = portable_path(receipt.get("receipt") or receipt.get("provenance") or "")
        expected = receipt.get("receipt_sha256") or receipt.get("provenance_sha256")
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Answer-review receipt changed or is missing: {name}")
    packet = list(read_jsonl(args.packet))
    final = list(read_jsonl(args.final_reviews))
    report = summarize(gold, packet, key, final)
    report["sample_flow"] = sample_flow(gold, gold_receipt)
    readers = manifest.get("reader_ids", [])
    if len(set(readers)) != 2:
        raise ValueError("Answer reviewer identities are invalid")
    for row in final:
        validate_compiled_resolution(row, manifest, readers, row["review_id"])
    report["gold_sha256"] = gold_receipt["gold_sha256"]
    report["execution_lock_sha256"] = lock_hash
    report["blank_packet_sha256"] = sha256(packet_bytes).hexdigest()
    report["final_review_sha256"] = manifest["final_sha256"]
    report["reader_agreement_counts"] = manifest.get("agreement_counts")
    report["answer_review_design"] = review_design(manifest)
    report["gold_review_design"] = review_design(gold_receipt)
    report["limitations"] = limitations(report["answer_review_design"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "accepted_items": len(gold),
                      "method_counts": report["method_counts"]}))


if __name__ == "__main__":
    main()
