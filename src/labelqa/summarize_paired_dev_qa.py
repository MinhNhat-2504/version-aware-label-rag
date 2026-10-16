"""Compare paired local QA diagnostics on frozen development prompts."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

from .data import read_jsonl
from .score_qa import score_one, validate_provenance


ROOT = Path("results")
METHODS = ("bm25", "learned_cover")


def paired_interval(rows: list[dict], key: str, draws: int = 2000) -> dict:
    """Paired bootstrap by label, counting one or more questions in a cluster."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["set_id"]].append((int(row["bm25"][key]), int(row["learned_cover"][key])))
    labels = sorted(groups)
    if not labels:
        raise ValueError("No paired rows")
    delta = sum(new - old for pairs in groups.values() for old, new in pairs) / len(rows)
    rng = random.Random(20261016)
    samples = []
    for _ in range(draws):
        chosen = rng.choices(labels, k=len(labels))
        pairs = [pair for label in chosen for pair in groups[label]]
        samples.append(sum(new - old for old, new in pairs) / len(pairs))
    samples.sort()
    return {"candidate_minus_bm25": round(delta, 4),
            "ci95_label_bootstrap": [round(samples[int(draws * .025)], 4),
                                     round(samples[min(int(draws * .975), draws - 1)], 4)],
            "n_questions": len(rows), "n_labels": len(labels), "draws": draws, "seed": 20261016}


def main() -> None:
    manifest_path = ROOT / "paired_dev_qa_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts, responses, scores = {}, {}, {}
    sources = [manifest_path]
    for method in METHODS:
        prompt_path = ROOT / f"paired_dev_qa_{method}_prompts.jsonl"
        response_path = ROOT / f"paired_dev_qa_{method}_responses.jsonl"
        if sha256(prompt_path.read_bytes()).hexdigest() != manifest["prompts_sha256"][method]:
            raise ValueError(f"Frozen {method} prompts changed")
        prompt_rows = list(read_jsonl(prompt_path))
        response_rows = list(read_jsonl(response_path))
        prompts[method] = {row["qid"]: row for row in prompt_rows}
        responses[method] = {row["qid"]: row for row in response_rows}
        if len(prompts[method]) != len(prompt_rows) or set(prompts[method]) != set(responses[method]):
            raise ValueError(f"Missing/duplicate {method} question or response")
        validate_provenance(prompts[method], responses[method])
        scores[method] = {qid: score_one(prompt, responses[method][qid])
                          for qid, prompt in prompts[method].items()}
        sources.extend((prompt_path, response_path))
    if prompts["bm25"].keys() != prompts["learned_cover"].keys():
        raise ValueError("Paired methods have different question sets")
    paired = []
    for item in manifest["sample"]:
        qid = item["qid"]
        row = {"qid": qid, "set_id": item["set_id"], "cohort": item["cohort"]}
        for method in METHODS:
            score = scores[method][qid]
            response = responses[method][qid]
            prompt = prompts[method][qid]
            valid = bool(score.get("valid_schema"))
            row[method] = {
                "retrieval_complete_upstream_gold": set(prompt["gold_indices"]) <= set(prompt["retrieved_indices"]),
                "schema_valid": valid,
                "non_abstaining": score.get("abstain") is False,
                "strict_mechanical_output": (valid and score.get("abstain") is False
                                             and score.get("invalid_citation_count") == 0
                                             and score.get("uncited_claim_count") == 0),
                "all_upstream_gold_ids_cited": valid and score.get("gold_citation_recall") == 1.0,
                "model_error": bool(response.get("error")),
                "prompt_tokens": response.get("prompt_eval_count"),
                "output_tokens": response.get("eval_count"),
                "duration_seconds": (response["total_duration_ns"] / 1e9
                                     if isinstance(response.get("total_duration_ns"), (int, float)) else None),
                "lexical_answer_f1": score.get("lexical_answer_f1"),
            }
        paired.append(row)
    cohorts = {}
    for cohort in dict.fromkeys(row["cohort"] for row in paired):
        subset = [row for row in paired if row["cohort"] == cohort]
        same_order = [row for row in subset if prompts["bm25"][row["qid"]]["retrieved_indices"]
                      == prompts["learned_cover"][row["qid"]]["retrieved_indices"]]
        both_complete = [row for row in subset if row["bm25"]["retrieval_complete_upstream_gold"]
                         and row["learned_cover"]["retrieval_complete_upstream_gold"]]
        methods = {}
        for method in METHODS:
            measurements = [row[method] for row in subset]
            durations = [value["duration_seconds"] for value in measurements if value["duration_seconds"] is not None]
            methods[method] = {
                "n": len(subset),
                **{key + "_count": sum(bool(value[key]) for value in measurements)
                   for key in ("retrieval_complete_upstream_gold", "schema_valid", "non_abstaining",
                               "strict_mechanical_output", "all_upstream_gold_ids_cited", "model_error")},
                "mean_duration_seconds": round(sum(durations) / len(durations), 2) if durations else None,
                "mean_prompt_tokens": round(sum(value["prompt_tokens"] for value in measurements
                                                if value["prompt_tokens"] is not None)
                                            / sum(value["prompt_tokens"] is not None for value in measurements), 1)
                if any(value["prompt_tokens"] is not None for value in measurements) else None,
            }
        cohorts[cohort] = {"methods": methods,
                           "diagnostics": {
                               "identical_ranked_passages": len(same_order),
                               "identical_passage_sets_any_order": sum(
                                   set(prompts["bm25"][row["qid"]]["retrieved_indices"])
                                   == set(prompts["learned_cover"][row["qid"]]["retrieved_indices"])
                                   for row in subset),
                               "identical_prompt_different_parsed_response": sum(
                                   responses["bm25"][row["qid"]].get("parsed_response")
                                   != responses["learned_cover"][row["qid"]].get("parsed_response")
                                   for row in same_order),
                               "both_methods_retrieved_all_gold": len(both_complete),
                               "strict_bm25_given_both_complete": sum(row["bm25"]["strict_mechanical_output"]
                                                                     for row in both_complete),
                               "strict_learned_given_both_complete": sum(
                                   row["learned_cover"]["strict_mechanical_output"] for row in both_complete),
                           },
                           "paired_retrieval_delta": paired_interval(subset, "retrieval_complete_upstream_gold"),
                           "paired_strict_output_delta": paired_interval(subset, "strict_mechanical_output")}
    report = {
        "status": "development_out_of_fold_generation_mechanical_diagnostics_only",
        "inputs_sha256": {str(path): sha256(path.read_bytes()).hexdigest() for path in sources},
        "cohorts": cohorts, "paired_items": paired,
        "limitations": [
            "This is a small development pilot; challenge strata were selected from retrieval gold.",
            "Output validity, citation IDs, lexical F1 and upstream gold matching do not establish clinical answer or claim support.",
            "One seeded local-model response per method/question; runtime variation can affect results.",
            "No external holdout or benchmark test question was run.",
        ],
    }
    output = ROOT / "paired_dev_qa_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cohorts": cohorts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
