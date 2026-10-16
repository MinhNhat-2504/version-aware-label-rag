"""Reanalyse the saved paired development generation pilot across both calls."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import portable_path
from .summarize_holdout_retrieval import paired_summary


SOURCE = Path("results/paired_dev_qa_four_method_comparison.json")
OUTPUT = Path("results/paired_dev_qa_repeat_sensitivity.json")
METHODS = ("bm25", "learned_cover", "dense", "bm25_dense_rrf")
COHORT = "representative_hash_sample"


def read_prompts(method: str, expected_hash: str) -> dict[str, dict]:
    path = Path(f"results/paired_dev_qa_{method}_prompts.jsonl")
    if sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Frozen prompt changed: {method}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len({row["qid"] for row in rows}):
        raise ValueError(f"Duplicate question in prompts: {method}")
    return {row["qid"]: row for row in rows}


def build() -> dict:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if source.get("status") != "four_method_two_call_development_diagnostic_not_clinical_answer_accuracy":
        raise ValueError("Unexpected development report")
    # Keys were written on Windows; compare them as POSIX paths on every platform.
    inputs = {portable_path(name).as_posix(): digest for name, digest in source["inputs_sha256"].items()}
    for name, digest in inputs.items():
        if sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen report input changed: {name}")
    prompts = {method: read_prompts(method, inputs[f"results/paired_dev_qa_{method}_prompts.jsonl"])
               for method in METHODS}
    rows = [row for row in source["items"] if row["cohort"] == COHORT]
    if len(rows) != 24 or len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("Representative sample differs from the frozen 24 items")
    if len({row["set_id"] for row in rows}) != len(rows):
        raise ValueError("Representative sample is not label-distinct")
    details = []
    for row in rows:
        qid = row["qid"]
        aligned = [prompts[method][qid] for method in METHODS]
        if len({(p["set_id"], p["question"], p["cohort"], tuple(p["gold_indices"]))
                for p in aligned}) != 1:
            raise ValueError(f"Question/gold changed across methods: {qid}")
        prompt_words = {method: len(prompts[method][qid]["user_prompt"].split())
                        for method in METHODS}
        passed = {method: {call: bool(row[method][call]["strict_mechanical_output"])
                           for call in ("initial", "repeat")} for method in METHODS}
        details.append({"qid": qid, "set_id": row["set_id"], "prompt_words": prompt_words,
                        "bm25_learned_identical_prompt":
                            all(prompts["bm25"][qid][key] == prompts["learned_cover"][qid][key]
                                for key in ("system_prompt", "user_prompt")),
                        "all_gold": {method: row[method]["retrieval_complete_upstream_gold"]
                                     for method in METHODS}, "mechanical_pass": passed})
    comparisons = {}
    for method in METHODS[1:]:
        comparisons[method] = {}
        for state in ("initial", "repeat", "both_calls"):
            items = []
            for row in details:
                def outcome(name: str) -> bool:
                    calls = row["mechanical_pass"][name]
                    return (calls["initial"] and calls["repeat"]) if state == "both_calls" else calls[state]
                items.append({"set_id": row["set_id"], "candidate": outcome(method),
                              "bm25": outcome("bm25")})
            comparisons[method][state] = paired_summary(items, "candidate", "bm25")
    method_counts = {}
    for method in METHODS:
        method_counts[method] = {
            "initial_pass": sum(row["mechanical_pass"][method]["initial"] for row in details),
            "repeat_pass": sum(row["mechanical_pass"][method]["repeat"] for row in details),
            "both_calls_pass": sum(all(row["mechanical_pass"][method].values()) for row in details),
            "decision_changed_between_calls": sum(
                row["mechanical_pass"][method]["initial"] != row["mechanical_pass"][method]["repeat"]
                for row in details),
            "mean_user_prompt_words": round(sum(row["prompt_words"][method] for row in details)
                                            / len(details), 1),
        }
    identical = [row for row in details if row["bm25_learned_identical_prompt"]]
    return {"status": "development_mechanical_endpoint_repeat_sensitivity_not_clinical_accuracy",
            "source_report_sha256": sha256(SOURCE.read_bytes()).hexdigest(),
            "n": len(details), "method_counts": method_counts, "paired_vs_bm25": comparisons,
            "bm25_learned_identical_prompt_count": len(identical),
            "identical_prompt_different_mechanical_decision_initial": sum(
                row["mechanical_pass"]["bm25"]["initial"] !=
                row["mechanical_pass"]["learned_cover"]["initial"] for row in identical),
            "identical_prompt_different_mechanical_decision_repeat": sum(
                row["mechanical_pass"]["bm25"]["repeat"] !=
                row["mechanical_pass"]["learned_cover"]["repeat"] for row in identical),
            "items": details,
            "limitations": ["Only the 24 hash-selected development questions contribute to rates.",
                            "The outcome tests response structure and citation IDs, not semantic support.",
                            "These calls share a model and seed but may differ because of runtime nondeterminism.",
                            "Bootstrap intervals are descriptive on a small development sample."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite report: {args.output}")
    report = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n": report["n"],
                      "counts": report["method_counts"]}))


if __name__ == "__main__":
    main()
