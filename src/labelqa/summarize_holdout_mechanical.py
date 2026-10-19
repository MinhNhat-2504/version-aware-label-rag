"""Descriptive mechanical answer scores on the holdout against the locked reference.

Under Amendment 6 there is no human answer review, so the answer endpoint is mechanical
only. Per item and method, ``mechanical_success`` requires a valid response schema, the
correct answer/refusal decision for the reference answerability, no citation outside the
retrieved passages, a citation on every claim and, for answerable items, cited passages
covering every reference evidence group. None of this checks whether a claim is true or
clinically adequate. Paired method differences use the label-pair bootstrap.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import portable_path
from .holdout_reporting import COHORTS, cohort_for, sample_flow
from .prepare_holdout_eval import DEFAULT_GOLD, load_manual_gold, model_weights
from .review_resolution import review_design
from .score_qa import score_one
from .summarize_holdout_retrieval import METHODS, paired_summary, read_prompts


STATUS = "descriptive_mechanical_answer_scores_against_reference_no_clinical_claim"
COMPONENTS = ("valid_schema", "correct_refusal_decision", "no_invalid_citation", "every_claim_cited",
              "cited_complete_evidence")


def components(prompt: dict, response: dict, answerable: bool) -> dict:
    score = score_one(prompt, response)
    valid = bool(score.get("valid_schema")) and not response.get("error")
    parts = {
        "valid_schema": valid,
        "correct_refusal_decision": valid and score.get("correct_refusal_decision") is True,
        "no_invalid_citation": valid and score.get("invalid_citation_count") == 0,
        "every_claim_cited": valid and score.get("uncited_claim_count", 0) == 0,
        "cited_complete_evidence": (valid and score.get("cited_complete_evidence") is True) if answerable else None,
    }
    parts["mechanical_success"] = all(value for value in parts.values() if value is not None)
    return parts


def summarize(gold: list[dict], prompts: dict[str, dict[str, dict]],
              responses: dict[str, dict[str, dict]]) -> dict:
    accepted = {row["qid"]: row for row in gold}
    if any(set(prompts[m]) != set(accepted) or set(responses[m]) != set(accepted) for m in METHODS):
        raise ValueError("Prompt/response QIDs differ from the reference items")
    per_item = []
    for qid, row in accepted.items():
        outcome = {"qid": qid, "set_id": row["set_id"], "spl_version": row["spl_version"],
                   "answerable": row["answerable"], "cohort": cohort_for(row["source_snapshot"])}
        for method in METHODS:
            prompt, response = prompts[method][qid], responses[method][qid]
            expected = sha256((prompt["system_prompt"] + prompt["user_prompt"]).encode()).hexdigest()
            if response.get("prompt_sha256") != expected:
                raise ValueError(f"Response was made for another prompt: {qid}, {method}")
            parts = components(prompt, response, row["answerable"])
            outcome[method] = parts["mechanical_success"]
            outcome[f"{method}_components"] = parts
        per_item.append(outcome)

    def counts(subset: list[dict]) -> dict:
        result = {}
        for method in METHODS:
            parts = [item[f"{method}_components"] for item in subset]
            result[method] = {"items": len(subset), "mechanical_success": sum(item[method] for item in subset),
                              **{name: sum(bool(p[name]) for p in parts if p[name] is not None)
                                 for name in COMPONENTS}}
        return result

    def compare(subset: list[dict]) -> dict | None:
        if not subset:
            return None
        return {"learned_vs_bm25": paired_summary(subset, "learned_cover", "bm25"),
                "learned_vs_ontology": paired_summary(subset, "learned_cover", "ontology_cover")}

    answerable = [item for item in per_item if item["answerable"]]
    by_cohort = {}
    for cohort in COHORTS.values():
        subset = [item for item in per_item if item["cohort"] == cohort]
        by_cohort[cohort] = {"items": len(subset), "method_counts": counts(subset), "paired_all": compare(subset)}
    return {"status": STATUS, "items": len(per_item), "answerable_items": len(answerable),
            "method_counts": counts(per_item), "paired_all": compare(per_item),
            "paired_answerable_only": compare(answerable), "by_cohort": by_cohort, "per_item": per_item,
            "limitations": ["Mechanical checks only: schema, refusal decision, citation validity and evidence "
                            "coverage. No claim is checked for truth, support or clinical adequacy.",
                            "Intervals are descriptive; the holdout is not a confirmatory test."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--execution-lock", type=Path, default=Path("paper/HOLDOUT_EXECUTION_LOCK.json"))
    parser.add_argument("--output", type=Path, default=Path("results/holdout_mechanical_scores.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a holdout report")
    gold, receipt = load_manual_gold(args.gold)  # Gate before any prompt or response is read.
    from .lock_holdout_execution import verify_lock
    from .prepare_blinded_answer_review import load_responses
    lock, lock_hash = verify_lock(args.execution_lock, gold, args.gold, receipt)
    gold_manifest_hash = sha256(args.gold.with_suffix(".manifest.json").read_bytes()).hexdigest()
    selector_hash = model_weights()[1]
    prompts, responses = {}, {}
    for method in METHODS:
        prompts[method] = read_prompts(portable_path(lock["prompt_paths"][method]), method, receipt["gold_sha256"],
                                       gold_manifest_hash, selector_hash if method == "learned_cover" else None,
                                       lock_hash)
        responses[method] = load_responses(portable_path(lock["response_paths"][method]), prompts[method])
        if any(row.get("model_digest") != lock["answer_model_digest"] or row.get("execution_lock_sha256") != lock_hash
               or row.get("decoding_options") != lock["decoding_options"] for row in responses[method].values()):
            raise ValueError(f"Answering model/settings differ from execution lock: {method}")
    report = summarize(gold, prompts, responses)
    report["sample_flow"] = sample_flow(gold, receipt)
    report["reference_review_design"] = review_design(receipt)
    report["reference_status"] = receipt["status"]
    report["gold_sha256"] = receipt["gold_sha256"]
    report["execution_lock_sha256"] = lock_hash
    report["response_sha256"] = {m: sha256(portable_path(lock["response_paths"][m]).read_bytes()).hexdigest()
                                 for m in METHODS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "items": report["items"],
                      "mechanical_success": {m: c["mechanical_success"] for m, c in report["method_counts"].items()}}))


if __name__ == "__main__":
    main()
