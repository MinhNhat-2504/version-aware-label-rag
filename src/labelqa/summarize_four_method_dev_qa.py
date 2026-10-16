"""Summarize four matched retrieval presentations and two local-model calls."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .score_qa import score_one, validate_provenance


ROOT = Path("results")
METHODS = ("bm25", "learned_cover", "dense", "bm25_dense_rrf")
RUNS = ("initial", "repeat")


def main() -> None:
    base_path = ROOT / "paired_dev_qa_manifest.json"
    dense_path = ROOT / "paired_dev_qa_dense_manifest.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    dense = json.loads(dense_path.read_text(encoding="utf-8"))
    if dense["sample_manifest_sha256"] != sha256(base_path.read_bytes()).hexdigest():
        raise ValueError("Dense prompts use a different sample")
    prompts, responses = {}, {}
    sources = [base_path, dense_path]
    for method in METHODS:
        prompt_path = ROOT / f"paired_dev_qa_{method}_prompts.jsonl"
        expected = (base if method in ("bm25", "learned_cover") else dense)["prompts_sha256"][method]
        if sha256(prompt_path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen prompts changed: {method}")
        prompt_rows = list(read_jsonl(prompt_path))
        prompts[method] = {row["qid"]: row for row in prompt_rows}
        if len(prompts[method]) != len(prompt_rows):
            raise ValueError(f"Duplicate prompt ID: {method}")
        sources.append(prompt_path)
        for run in RUNS:
            suffix = "" if run == "initial" else "_repeat"
            path = ROOT / f"paired_dev_qa_{method}{suffix}_responses.jsonl"
            response_rows = list(read_jsonl(path))
            responses[method, run] = {row["qid"]: row for row in response_rows}
            if len(response_rows) != len(responses[method, run]) or set(prompts[method]) != set(responses[method, run]):
                raise ValueError(f"Incomplete/duplicate responses: {method} {run}")
            validate_provenance(prompts[method], responses[method, run])
            sources.append(path)
    expected_ids = {row["qid"] for row in base["sample"]}
    if any(set(prompts[method]) != expected_ids for method in METHODS):
        raise ValueError("Methods use different development questions")
    items = []
    for sampled in base["sample"]:
        qid = sampled["qid"]
        reference = prompts["bm25"][qid]
        row = {"qid": qid, "set_id": sampled["set_id"], "cohort": sampled["cohort"]}
        for method in METHODS:
            prompt = prompts[method][qid]
            if ((prompt["set_id"], prompt["question"], prompt["gold_indices"], prompt["task"])
                    != (reference["set_id"], reference["question"], reference["gold_indices"], reference["task"])):
                raise ValueError(f"Question identity differs: {method} {qid}")
            retrieval = set(prompt["gold_indices"]) <= set(prompt["retrieved_indices"])
            calls = []
            for run in RUNS:
                response = responses[method, run][qid]
                comparator = responses["bm25", run][qid]
                if (response.get("model_digest") != comparator.get("model_digest")
                        or response.get("decoding_options") != comparator.get("decoding_options")):
                    raise ValueError(f"Model/decoding differs: {method} {run} {qid}")
                score = score_one(prompt, response)
                valid = bool(score.get("valid_schema"))
                calls.append({"schema_valid": valid,
                              "non_abstaining": score.get("abstain") is False,
                              "strict_mechanical_output": (valid and score.get("abstain") is False
                                                           and score.get("invalid_citation_count") == 0
                                                           and score.get("uncited_claim_count") == 0),
                              "all_upstream_gold_ids_cited": valid and score.get("gold_citation_recall") == 1.0,
                              "model_error": bool(response.get("error")),
                              "parsed_response": response.get("parsed_response")})
            row[method] = {"retrieval_complete_upstream_gold": retrieval,
                           "initial": {key: value for key, value in calls[0].items() if key != "parsed_response"},
                           "repeat": {key: value for key, value in calls[1].items() if key != "parsed_response"},
                           "exact_response_changed": calls[0]["parsed_response"] != calls[1]["parsed_response"]}
        items.append(row)
    cohorts = {}
    for cohort in dict.fromkeys(row["cohort"] for row in items):
        selected = [row for row in items if row["cohort"] == cohort]
        cohorts[cohort] = {method: {
            "n": len(selected),
            "mean_user_prompt_words": round(sum(len(prompts[method][row["qid"]]["user_prompt"].split())
                                                for row in selected) / len(selected), 1),
            "retrieval_complete_upstream_gold": sum(row[method]["retrieval_complete_upstream_gold"]
                                                    for row in selected),
            "exact_response_changed_between_calls": sum(row[method]["exact_response_changed"]
                                                        for row in selected),
            **{run: {**{key: sum(row[method][run][key] for row in selected)
                        for key in ("schema_valid", "non_abstaining", "strict_mechanical_output",
                                    "all_upstream_gold_ids_cited", "model_error")},
                     "mean_prompt_eval_tokens": round(
                         sum(responses[method, run][row["qid"]]["prompt_eval_count"] for row in selected
                             if responses[method, run][row["qid"]].get("prompt_eval_count") is not None)
                         / sum(responses[method, run][row["qid"]].get("prompt_eval_count") is not None
                               for row in selected), 1)
                     if any(responses[method, run][row["qid"]].get("prompt_eval_count") is not None
                            for row in selected) else None}
               for run in RUNS},
        } for method in METHODS}
    report = {
        "status": "four_method_two_call_development_diagnostic_not_clinical_answer_accuracy",
        "inputs_sha256": {str(path): sha256(path.read_bytes()).hexdigest() for path in sources},
        "cohorts": cohorts, "items": items,
        "limitations": ["Small development sample, with challenge cohorts selected on upstream gold outcomes.",
                        "Two local-model calls cannot estimate stable per-method probabilities.",
                        "Output format, passage IDs and upstream gold do not verify actual claim support.",
                        "No external holdout or benchmark test question was evaluated."],
    }
    output = ROOT / "paired_dev_qa_four_method_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "representative": cohorts["representative_hash_sample"]}))


if __name__ == "__main__":
    main()
