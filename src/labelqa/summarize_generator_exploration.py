"""Exploratory comparison of answering models and repeated runs on the locked holdout prompts.

Each run is given as ``name=model=prefix=mechanical_report``: responses are read from
``<prefix>_<method>_responses.jsonl`` and per-item outcomes from the mechanical report
(``labelqa.summarize_holdout_mechanical``) that verified that run's own execution lock.
The report gives per-run counts, a paired larger-minus-smaller model contrast with the
label-pair bootstrap, and run-to-run repeatability per model. Everything is descriptive.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .summarize_holdout_retrieval import METHODS, paired_summary


def load_run(spec: str) -> dict:
    name, model, prefix, report_path = spec.split("=")
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    responses = {}
    for method in METHODS:
        path = Path(f"{prefix}_{method}_responses.jsonl")
        responses[method] = {row["qid"]: row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    return {"name": name, "model": model, "report": report, "responses": responses,
            "report_sha256": sha256(Path(report_path).read_bytes()).hexdigest()}


def empty_answer(row: dict) -> bool:
    parsed = row.get("parsed_response")
    return not row.get("error") and isinstance(parsed, dict) and not str(parsed.get("answer", "")).strip()


def run_counts(run: dict) -> dict:
    counts = {}
    for method in METHODS:
        method_counts = run["report"]["method_counts"][method]
        rows = run["responses"][method].values()
        counts[method] = {"mechanical_success": method_counts["mechanical_success"],
                          "valid_schema": method_counts["valid_schema"],
                          "correct_refusal_decision": method_counts["correct_refusal_decision"],
                          "empty_answer_field": sum(empty_answer(row) for row in rows),
                          "generation_or_json_error": sum(bool(row.get("error")) for row in rows)}
    return counts


def size_contrast(small: dict, large: dict) -> dict:
    small_items = {item["qid"]: item for item in small["report"]["per_item"]}
    contrast = {}
    for method in METHODS:
        paired = [{"set_id": item["set_id"], "large": item[method], "small": small_items[item["qid"]][method]}
                  for item in large["report"]["per_item"]]
        contrast[method] = paired_summary(paired, "large", "small")
    return {"larger": large["name"], "smaller": small["name"], "mechanical_success": contrast}


def repeatability(runs: list[dict]) -> dict:
    result = {}
    for method in METHODS:
        outcomes = defaultdict(list)
        texts = defaultdict(list)
        for run in runs:
            for item in run["report"]["per_item"]:
                outcomes[item["qid"]].append(item[method])
            for qid, row in run["responses"][method].items():
                texts[qid].append(json.dumps(row.get("parsed_response"), sort_keys=True) if not row.get("error")
                                  else "ERROR:" + row.get("raw_response", ""))
        result[method] = {
            "runs": len(runs), "items": len(outcomes),
            "identical_pass_fail_all_runs": sum(len(set(v)) == 1 for v in outcomes.values()),
            "identical_parsed_response_all_runs": sum(len(set(v)) == 1 for v in texts.values()),
            "mechanical_success_by_run": [run["report"]["method_counts"][method]["mechanical_success"]
                                          for run in runs],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="name=model=prefix=mechanical_report")
    parser.add_argument("--smaller", required=True, help="Run name of the smaller model's reference run")
    parser.add_argument("--larger", required=True, help="Run name of the larger model's first run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an exploratory report")
    runs = [load_run(spec) for spec in args.run]
    by_name = {run["name"]: run for run in runs}
    by_model = defaultdict(list)
    for run in runs:
        by_model[run["model"]].append(run)
    reference = {run["report"]["gold_sha256"] for run in runs}
    if len(reference) != 1:
        raise ValueError("Runs were scored against different references")
    report = {
        "status": "exploratory_generator_size_and_repeatability_no_clinical_claim",
        "plan": "paper/EXPLORATORY_PLAN_GENERATORS.md",
        "plan_sha256": sha256(Path("paper/EXPLORATORY_PLAN_GENERATORS.md").read_bytes()).hexdigest(),
        "reference_sha256": reference.pop(),
        "runs": {run["name"]: {"model": run["model"], "lock_sha256": run["report"]["execution_lock_sha256"],
                               "mechanical_report_sha256": run["report_sha256"], "counts": run_counts(run)}
                 for run in runs},
        "size_contrast": size_contrast(by_name[args.smaller], by_name[args.larger]),
        "repeatability": {model: repeatability(model_runs) for model, model_runs in by_model.items()
                          if len(model_runs) > 1},
        "limitations": ["Exploratory and descriptive; mechanical checks only against an AI silver standard.",
                        "Repeated runs use identical prompts and settings; differences reflect runtime nondeterminism."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "runs": list(report["runs"]),
                      "success": {name: {m: c["mechanical_success"] for m, c in run["counts"].items()}
                                  for name, run in report["runs"].items()}}))


if __name__ == "__main__":
    main()
