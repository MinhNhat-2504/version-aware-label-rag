"""Quantify fixed-prompt local-model variation in two paired development runs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .score_qa import score_one, validate_provenance


ROOT = Path("results")
METHODS = ("bm25", "learned_cover")
RUNS = ("initial", "repeat")


def response_path(method: str, run: str) -> Path:
    suffix = "" if run == "initial" else "_repeat"
    return ROOT / f"paired_dev_qa_{method}{suffix}_responses.jsonl"


def main() -> None:
    manifest_path = ROOT / "paired_dev_qa_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prompts = {}
    responses = {}
    inputs = [manifest_path]
    for method in METHODS:
        path = ROOT / f"paired_dev_qa_{method}_prompts.jsonl"
        if sha256(path.read_bytes()).hexdigest() != manifest["prompts_sha256"][method]:
            raise ValueError(f"Frozen prompts changed for {method}")
        prompts[method] = {row["qid"]: row for row in read_jsonl(path)}
        inputs.append(path)
        for run in RUNS:
            result_path = response_path(method, run)
            rows = list(read_jsonl(result_path))
            responses[method, run] = {row["qid"]: row for row in rows}
            if len(rows) != len(responses[method, run]) or set(prompts[method]) != set(responses[method, run]):
                raise ValueError(f"Incomplete/duplicate {method} {run} responses")
            validate_provenance(prompts[method], responses[method, run])
            inputs.append(result_path)
    items = []
    for item in manifest["sample"]:
        qid = item["qid"]
        row = {"qid": qid, "set_id": item["set_id"], "cohort": item["cohort"]}
        for run in RUNS:
            baseline = responses["bm25", run][qid]
            candidate = responses["learned_cover", run][qid]
            if (baseline.get("model_digest") != candidate.get("model_digest")
                    or baseline.get("decoding_options") != candidate.get("decoding_options")):
                raise ValueError(f"Paired model or decoding settings differ: {qid} {run}")
        for method in METHODS:
            entries = []
            for run in RUNS:
                response = responses[method, run][qid]
                score = score_one(prompts[method][qid], response)
                valid = bool(score.get("valid_schema"))
                entries.append({
                    "run": run, "schema_valid": valid, "abstain": score.get("abstain"),
                    "strict_mechanical_output": (valid and score.get("abstain") is False
                                                 and score.get("invalid_citation_count") == 0
                                                 and score.get("uncited_claim_count") == 0),
                    "parsed_response": response.get("parsed_response"),
                    "model_error": bool(response.get("error")),
                    "model_digest": response.get("model_digest"),
                    "decoding_options": response.get("decoding_options"),
                })
            if entries[0]["model_digest"] != entries[1]["model_digest"] or entries[0]["decoding_options"] != entries[1]["decoding_options"]:
                raise ValueError(f"Model or decoding settings changed: {method} {qid}")
            row[method] = {
                "strict_initial": entries[0]["strict_mechanical_output"],
                "strict_repeat": entries[1]["strict_mechanical_output"],
                "exact_response_changed": entries[0]["parsed_response"] != entries[1]["parsed_response"],
                "schema_changed": entries[0]["schema_valid"] != entries[1]["schema_valid"],
                "abstention_changed": entries[0]["abstain"] != entries[1]["abstain"],
                "model_error_initial": entries[0]["model_error"],
                "model_error_repeat": entries[1]["model_error"],
            }
        row["identical_prompt_across_methods"] = (
            prompts["bm25"][qid]["system_prompt"] == prompts["learned_cover"][qid]["system_prompt"]
            and prompts["bm25"][qid]["user_prompt"] == prompts["learned_cover"][qid]["user_prompt"])
        items.append(row)
    cohorts = {}
    for cohort in dict.fromkeys(row["cohort"] for row in items):
        selected = [row for row in items if row["cohort"] == cohort]
        cohorts[cohort] = {
            "n": len(selected),
            "identical_prompts_across_methods": sum(row["identical_prompt_across_methods"] for row in selected),
            "methods": {method: {
                "strict_initial": sum(row[method]["strict_initial"] for row in selected),
                "strict_repeat": sum(row[method]["strict_repeat"] for row in selected),
                "exact_response_changed": sum(row[method]["exact_response_changed"] for row in selected),
                "schema_changed": sum(row[method]["schema_changed"] for row in selected),
                "abstention_changed": sum(row[method]["abstention_changed"] for row in selected),
                "model_error_initial": sum(row[method]["model_error_initial"] for row in selected),
                "model_error_repeat": sum(row[method]["model_error_repeat"] for row in selected),
            } for method in METHODS},
        }
    report = {
        "status": "same_prompt_seed_model_repeatability_development_only",
        "run_order": {"initial": list(METHODS), "repeat": list(reversed(METHODS))},
        "inputs_sha256": {str(path): sha256(path.read_bytes()).hexdigest() for path in inputs},
        "cohorts": cohorts, "items": items,
        "limitations": ["Two calls per method/question cannot estimate a stable error probability.",
                        "The local model is run in sequential blocks; exact outputs can depend on runtime state.",
                        "Mechanical output validity is not clinical answer or citation correctness."],
    }
    output = ROOT / "paired_dev_qa_repeatability.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cohorts": cohorts}))


if __name__ == "__main__":
    main()
