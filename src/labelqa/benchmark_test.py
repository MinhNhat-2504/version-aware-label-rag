"""Preregister, then evaluate exactly once, the frozen retrieval methods on FDARxBench test.

`lock` fingerprints the benchmark snapshot, selector model, retrieval and scoring code,
the endpoint, contrasts, bootstrap settings and decision rule before any test retrieval
runs. It reads test question metadata only to fix the denominator. `evaluate` refuses to
run without an unchanged lock or when a test report already exists, so the confirmatory
analysis cannot be silently repeated or retuned.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .data import Question, load_labels, load_questions, split_label, validate_data
from .evaluate import paired_cluster_bootstrap, summarize
from .learned_retrieval import retrieve as learned_retrieve
from .plan_holdout_sensitivity import optimistic_exact_power
from .prepare_holdout_eval import MODEL_REPORT, model_weights
from .retrieval import LabelIndex


SOURCE = Path("data/upstream/FDARxBench/data")
LOCK = Path("paper/BENCHMARK_TEST_LOCK.json")
OUTPUT = Path("results/benchmark_test_retrieval.json")
PREDICTIONS = Path("results/benchmark_test_predictions.jsonl")
CODE_FILES = ("data.py", "retrieval.py", "learned_retrieval.py", "evaluate.py",
              "prepare_holdout_eval.py", "benchmark_test.py")
PRIMARY = ("learned_cover", "bm25")
SECONDARY = ("learned_cover", "ontology_cover")
EXPLORATORY_METHODS = ("tfidf", "rrf_lexical")
METHODS = ("bm25", "ontology_cover", "learned_cover") + EXPLORATORY_METHODS
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261018
STATUS = "fdarxbench_test_preregistered_before_any_test_retrieval"
# Discordant cells observed before the lock; used for planning only.
PLANNING_EFFECTS = {
    "dev_out_of_fold": {"learned_only": 175, "bm25_only": 91, "n": 2393},
    "validation_inspected": {"learned_only": 31, "bm25_only": 18, "n": 559},
}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def eligible_test_questions(questions: list[Question]) -> list[Question]:
    return [q for q in questions if split_label(q.set_id) == "test" and q.task == "multihop"
            and 0 < len(q.gold) <= 5]


def decision(interval: list[float]) -> str:
    """Prespecified three-way reading of a 95% label-bootstrap interval."""
    low, high = interval
    if low > 0:
        return "candidate_superior_on_retrieval_proxy"
    if high < 0:
        return "candidate_inferior_on_retrieval_proxy"
    return "inconclusive_interval_includes_zero"


def planning_power(n: int) -> dict:
    rows = {}
    for name, cells in PLANNING_EFFECTS.items():
        discordant = cells["learned_only"] + cells["bm25_only"]
        rate, share = discordant / cells["n"], cells["learned_only"] / discordant
        rows[name] = {**cells, "discordance_rate": round(rate, 4),
                      "implied_difference": round(rate * (2 * share - 1), 4),
                      "optimistic_power_at_test_n": round(optimistic_exact_power(n, rate, share), 4)}
    return rows


def lock_payload(questions: list[Question], code_root: Path) -> dict:
    eligible = eligible_test_questions(questions)
    if not eligible:
        raise ValueError("No eligible test questions")
    selector_hash = model_weights()[1]
    return {
        "status": STATUS,
        "benchmark_sha256": {"labels": digest(SOURCE / "labels/labels.jsonl"),
                             "qa": digest(SOURCE / "qa/qa.jsonl")},
        "selector_model_sha256": selector_hash,
        "selector_validation_report_sha256": digest(MODEL_REPORT),
        "code_sha256": {name: digest(code_root / name) for name in CODE_FILES},
        "split_rule": "SHA-256 set_id buckets 85-99 (data.split_label == 'test')",
        "endpoint": ("all_gold@5: every upstream context.doc_chunk_index gold passage is in the "
                     "top five; multihop questions with 1 to 5 gold indices"),
        "denominator": {"questions": len(eligible), "labels": len({q.set_id for q in eligible}),
                        "question_ids_sha256": sha256("\n".join(sorted(q.qid for q in eligible))
                                                      .encode()).hexdigest()},
        "methods": list(METHODS), "passage_budget": 5,
        "primary_contrast": f"{PRIMARY[0]} minus {PRIMARY[1]}",
        "secondary_contrast": f"{SECONDARY[0]} minus {SECONDARY[1]}",
        "exploratory_methods": list(EXPLORATORY_METHODS),
        "bootstrap": {"unit": "label set_id", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "interval": "95% percentile"},
        "decision_rule": ("Superior only if the primary interval lower bound exceeds 0; inferior if its "
                          "upper bound is below 0; otherwise inconclusive. No threshold, weight or "
                          "method change after opening; the secondary contrast is reported with the "
                          "same rule but cannot rescue the primary."),
        "planning_power_item_independent": planning_power(len(eligible)),
        "limitations": [
            "Upstream gold indices omit unadjudicated equivalent passages; this is a retrieval proxy.",
            "No answer generation, claim support or clinical correctness is measured.",
            "Benchmark questions were generated with LLM assistance and may not reflect prospective use.",
        ],
        "test_retrieval_outputs_read": 0,
    }


def verify_lock(path: Path, questions: list[Question], code_root: Path) -> tuple[dict, str]:
    if not path.is_file():
        raise ValueError("Benchmark test lock is required before any test retrieval")
    locked = json.loads(path.read_text(encoding="utf-8"))
    expected = lock_payload(questions, code_root)
    if {key: value for key, value in locked.items() if key != "created_utc"} != expected:
        raise ValueError("Benchmark, selector, code or preregistered analysis changed after lock")
    datetime.fromisoformat(locked["created_utc"])
    return locked, digest(path)


def paired_cells(baseline, candidate) -> dict[str, int]:
    cells = {"both": 0, "candidate_only": 0, "baseline_only": 0, "neither": 0}
    for (question, base), (_, new) in zip(baseline, candidate):
        gold = set(question.gold)
        a, b = gold <= set(new), gold <= set(base)
        cells["both" if a and b else "candidate_only" if a else "baseline_only" if b else "neither"] += 1
    return cells


def evaluate(questions: list[Question], labels: dict, locked: dict, lock_hash: str) -> tuple[dict, list[dict]]:
    eligible = eligible_test_questions(questions)
    weights = model_weights()[0]
    indexes: dict[str, LabelIndex] = {}
    outcomes: dict[str, list] = {method: [] for method in METHODS}
    predictions = []
    for question in eligible:
        index = indexes.setdefault(question.set_id, LabelIndex(labels[question.set_id]))
        rankings = {method: (learned_retrieve(index, question.text, weights, k=5, coverage=True)
                             if method == "learned_cover" else index.retrieve(question.text, method, 5))
                    for method in METHODS}
        for method, ranking in rankings.items():
            if len(ranking) > 5 or len(set(ranking)) != len(ranking):
                raise ValueError(f"Invalid passage budget: {question.qid}, {method}")
            outcomes[method].append((question, ranking))
        predictions.append({"qid": question.qid, "set_id": question.set_id,
                            "gold_indices": list(question.gold), "rankings": rankings})

    def contrast(candidate: str, baseline: str) -> dict:
        result = paired_cluster_bootstrap(outcomes[baseline], outcomes[candidate],
                                          draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
        return {**result, "paired_cells": paired_cells(outcomes[baseline], outcomes[candidate]),
                "decision": decision(result["ci95_label_bootstrap"])}

    report = {
        "status": "fdarxbench_test_evaluated_once_after_preregistration",
        "lock_sha256": lock_hash,
        "metrics": {method: summarize(rows)["multihop"] for method, rows in outcomes.items()},
        "primary": contrast(*PRIMARY),
        "secondary": contrast(*SECONDARY),
        "exploratory": {f"{method}_minus_bm25": contrast(method, "bm25") for method in EXPLORATORY_METHODS},
        "limitations": locked["limitations"],
    }
    return report, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("lock", "evaluate"))
    args = parser.parse_args()
    code_root = Path(__file__).resolve().parent
    labels = load_labels(SOURCE / "labels/labels.jsonl")
    questions = load_questions(SOURCE / "qa/qa.jsonl")
    validate_data(labels, questions)
    if OUTPUT.exists() or PREDICTIONS.exists():
        parser.error("The benchmark test has already been evaluated; it cannot be repeated")
    if args.action == "lock":
        if LOCK.exists():
            parser.error("Benchmark test lock already exists")
        payload = lock_payload(questions, code_root)
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"lock": str(LOCK), **payload["denominator"],
                          "planning_power": {name: row["optimistic_power_at_test_n"] for name, row
                                             in payload["planning_power_item_independent"].items()}}))
        return
    locked, lock_hash = verify_lock(LOCK, questions, code_root)
    report, predictions = evaluate(questions, labels, locked, lock_hash)
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    report["predictions_sha256"] = digest(PREDICTIONS)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT),
                      "primary": {key: report["primary"][key] for key in
                                  ("candidate_minus_baseline", "ci95_label_bootstrap", "decision")}}))


if __name__ == "__main__":
    main()
