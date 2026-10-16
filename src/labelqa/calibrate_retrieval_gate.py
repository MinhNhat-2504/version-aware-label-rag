"""Calibrate a retrieval-only abstention proxy from grouped dev crossfit predictions."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import random

from .data import load_labels, load_questions, read_jsonl, split_label
from .retrieval import CLAUSE_BOUNDARY, SECTION_INTENTS, LabelIndex, tokenize


SOURCE = Path("data/upstream/FDARxBench/data")
DEV_PREDICTIONS = Path("results/dev_learned_crossfit_predictions.jsonl")
VALIDATION_PREDICTIONS = Path("results/validation_learned_predictions.jsonl")
FEATURE_NAMES = ("selected_query_coverage", "top1_bm25_relative", "weakest_selected_bm25_relative",
                 "clause_coverage", "heading_intent_coverage", "distinct_section_fraction",
                 "question_length")
TARGET_DEV_RISK = 0.25
SEED = 20261015


def features(index: LabelIndex, question: str, ranking: list[int]) -> tuple[float, ...]:
    bm25 = index.bm25_scores(question)
    maximum = max(bm25, default=0.0) or 1.0
    query = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query) or 1.0
    covered = set().union(*(set(index.terms[i]) for i in ranking)) if ranking else set()
    clauses = [part for part in CLAUSE_BOUNDARY.split(question) if len(tokenize(part)) >= 2]
    if len(clauses) > 1:
        clause_ratios = []
        for clause in clauses[:3]:
            scores = index.bm25_scores(clause)
            clause_ratios.append(max((scores[i] for i in ranking), default=0.0)
                                 / (max(scores, default=0.0) or 1.0))
        clause_coverage = min(clause_ratios)
    else:
        clause_coverage = 1.0
    intents = [headings for trigger, headings in SECTION_INTENTS if trigger.search(question)]
    intent_coverage = (sum(any(any(heading in index.title_lines[i] for heading in headings)
                               for i in ranking) for headings in intents) / len(intents)) if intents else 1.0
    return (
        sum(index.idf.get(term, 0.0) for term in query & covered) / denominator,
        bm25[ranking[0]] / maximum if ranking else 0.0,
        min((bm25[i] / maximum for i in ranking), default=0.0),
        clause_coverage,
        intent_coverage,
        len({index.title_lines[i] for i in ranking}) / max(len(ranking), 1),
        min(len(tokenize(question)) / 20, 1.0),
    )


def probability(weights: list[float], values: tuple[float, ...]) -> float:
    z = max(-40.0, min(40.0, weights[0] + sum(w * x for w, x in zip(weights[1:], values))))
    return 1.0 / (1.0 + math.exp(-z))


def fit(rows: list[tuple[tuple[float, ...], bool]]) -> list[float]:
    weights = [0.0] * (len(FEATURE_NAMES) + 1)
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    for epoch in range(20):
        rng.shuffle(order)
        rate = 0.05 / (1 + 0.25 * epoch)
        for i in order:
            values, label = rows[i]
            error = float(label) - probability(weights, values)
            weights[0] += rate * error
            for j, value in enumerate(values, 1):
                weights[j] += rate * (error * value - 0.0001 * weights[j])
    return weights


def selective(scores: list[tuple[float, bool]], threshold: float) -> dict:
    chosen = [correct for score, correct in scores if score >= threshold]
    return {"accepted": len(chosen), "total": len(scores),
            "coverage": round(len(chosen) / len(scores), 4) if scores else 0.0,
            "observed_missing_gold_risk": round(1 - sum(chosen) / len(chosen), 4) if chosen else None}


def choose_threshold(scores: list[tuple[float, bool]], target: float) -> float:
    # Distinct thresholds avoid silently admitting extra tied predictions.
    eligible = [(len([1 for value, _ in scores if value >= threshold]), threshold)
                for threshold in {value for value, _ in scores}
                if (result := selective(scores, threshold))["accepted"] >= 20
                and result["observed_missing_gold_risk"] <= target]
    return max(eligible)[1] if eligible else 1.01


def main() -> None:
    labels = load_labels(SOURCE / "labels/labels.jsonl")
    questions = {q.qid: q for q in load_questions(SOURCE / "qa/qa.jsonl")}
    dev_predictions = list(read_jsonl(DEV_PREDICTIONS))
    validation_predictions = list(read_jsonl(VALIDATION_PREDICTIONS))
    indexes = {}

    def build(rows, split):
        examples = []
        for row in rows:
            q = questions[row["qid"]]
            if split_label(q.set_id) != split or q.task != "multihop" or not (0 < len(q.gold) <= 5):
                raise ValueError(f"Prediction entered wrong split: {q.qid}")
            if tuple(row["gold_indices"]) != q.gold:
                raise ValueError(f"Upstream gold changed for {q.qid}")
            if q.set_id not in indexes:
                indexes[q.set_id] = LabelIndex(labels[q.set_id])
            ranking = row["rankings"]["learned_cover"]
            if len(ranking) > 5 or len(set(ranking)) != len(ranking):
                raise ValueError(f"Invalid retrieval budget for {q.qid}")
            values = features(indexes[q.set_id], q.text, ranking)
            correct = set(q.gold) <= set(ranking)
            examples.append((q.qid, values, correct))
        if len({qid for qid, _, _ in examples}) != len(examples):
            raise ValueError(f"Duplicate prediction in {split}")
        return examples

    dev = build(dev_predictions, "dev")
    validation = build(validation_predictions, "validation")
    weights = fit([(values, correct) for _, values, correct in dev])
    dev_scores = [(probability(weights, values), correct) for _, values, correct in dev]
    validation_scores = [(probability(weights, values), correct) for _, values, correct in validation]
    threshold = choose_threshold(dev_scores, TARGET_DEV_RISK)
    model = {"fit_split": "dev grouped crossfit outputs only", "features": list(FEATURE_NAMES),
             "weights": [round(value, 8) for value in weights], "threshold": threshold,
             "target_empirical_dev_risk": TARGET_DEV_RISK, "seed": SEED,
             "dev_predictions_sha256": sha256(DEV_PREDICTIONS.read_bytes()).hexdigest(),
             "validation_predictions_sha256": sha256(VALIDATION_PREDICTIONS.read_bytes()).hexdigest()}
    model_path = Path("results/retrieval_gate_model.json")
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"status": "retrieval_gold_proxy_only_not_answer_safety_calibration",
              "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
              "threshold_selected_on_dev_only": threshold,
              "dev": selective(dev_scores, threshold),
              "validation": selective(validation_scores, threshold),
              "validation_curve": {str(t): selective(validation_scores, t)
                                   for t in (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)},
              "limitations": [
                  "Outcome is retrieval of all upstream gold indices, not claim support or clinical answer correctness.",
                  "Threshold satisfies an empirical dev criterion only; it is not a statistical safety guarantee.",
                  "Validation is assessed once without changing the threshold; sealed test and new-label holdout remain unopened.",
              ]}
    report_path = Path("results/retrieval_gate_validation.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "threshold": threshold,
                      "dev": report["dev"], "validation": report["validation"]}))


if __name__ == "__main__":
    main()
