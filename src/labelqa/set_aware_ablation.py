"""Evaluate a structured top-k update for evidence-set selection over the deep pool."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import random

from .crossfit_learned import FOLDS, fold_of
from .data import load_labels, load_questions, split_label
from .deep_pool_ablation import (
    DEV_PREDICTIONS,
    POOL_SIZE,
    VALIDATION_PREDICTIONS,
    deep_feature_pool,
    paired_cells,
    prediction_map,
)
from .evaluate import paired_cluster_bootstrap, summarize
from .learned_retrieval import FEATURE_NAMES, score, train
from .retrieval import LabelIndex, tokenize


SOURCE = Path("data/upstream/FDARxBench/data")
OUTPUT = Path("results/set_aware_ablation.json")
SET_EPOCHS = 4
SET_LEARNING_RATE = 0.0001
SEED = 20261017


def select_topk(index: LabelIndex, question: str, features: dict[int, tuple[float, ...]],
                pool: list[int], weights: list[float], k: int = 5) -> list[int]:
    """Use the frozen marginal-coverage objective on precomputed candidates."""
    query_terms = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query_terms) or 1.0
    selected: list[int] = []
    covered: set[str] = set()
    titles: set[str] = set()
    remaining = pool.copy()
    while remaining and len(selected) < k:
        def value(candidate: int) -> float:
            new_terms = (query_terms & set(index.terms[candidate])) - covered
            novelty = sum(index.idf.get(term, 0.0) for term in new_terms) / denominator
            diversity = float(index.title_lines[candidate] not in titles)
            return score(weights, features[candidate]) + 0.12 * novelty + 0.04 * diversity

        best = max(remaining, key=lambda candidate: (value(candidate), -candidate))
        selected.append(best)
        covered.update(query_terms & set(index.terms[best]))
        titles.add(index.title_lines[best])
        remaining.remove(best)
    return selected


def structured_difference(features: dict[int, tuple[float, ...]],
                          missing_gold: list[int], false_selected: list[int]) -> list[float]:
    """Mean missing-gold features minus mean false-selected features."""
    if not missing_gold or not false_selected:
        return [0.0] * len(FEATURE_NAMES)
    return [
        sum(features[index][column] for index in missing_gold) / len(missing_gold)
        - sum(features[index][column] for index in false_selected) / len(false_selected)
        for column in range(len(FEATURE_NAMES))
    ]


def train_set_aware(records: list[tuple[LabelIndex, str, dict[int, tuple[float, ...]],
                                        list[int], set[int]]],
                    initial_weights: list[float]) -> tuple[list[float], dict]:
    """Fine-tune passage weights only on complete-pool top-five set errors."""
    weights = initial_weights.copy()
    eligible = [record for record in records if record[4] <= set(record[3])]
    order = list(range(len(eligible)))
    generator = random.Random(SEED)
    updates_by_epoch = []
    for epoch in range(SET_EPOCHS):
        generator.shuffle(order)
        rate = SET_LEARNING_RATE / (1 + 0.25 * epoch)
        updates = 0
        for position in order:
            index, question, features, pool, gold = eligible[position]
            selected = select_topk(index, question, features, pool, weights)
            missing = sorted(gold - set(selected))
            false_selected = [candidate for candidate in selected if candidate not in gold]
            if not missing or not false_selected:
                continue
            difference = structured_difference(features, missing, false_selected)
            for column, value in enumerate(difference):
                weights[column] += rate * (value - 0.0001 * weights[column])
            updates += 1
        updates_by_epoch.append(updates)
    return weights, {
        "epochs": SET_EPOCHS,
        "learning_rate": SET_LEARNING_RATE,
        "seed": SEED,
        "complete_pool_training_questions": len(eligible),
        "updates_by_epoch": updates_by_epoch,
        "total_updates": sum(updates_by_epoch),
    }


def main() -> None:
    label_path = SOURCE / "labels/labels.jsonl"
    qa_path = SOURCE / "qa/qa.jsonl"
    labels = load_labels(label_path)
    questions = load_questions(qa_path)
    dev = [q for q in questions if split_label(q.set_id) == "dev" and q.task == "multihop"
           and 0 < len(q.gold) <= 5]
    validation = [q for q in questions if split_label(q.set_id) == "validation"
                  and q.task == "multihop" and 0 < len(q.gold) <= 5]
    indexes = {set_id: LabelIndex(labels[set_id])
               for set_id in {q.set_id for q in dev + validation}}
    dev_predictions = prediction_map(DEV_PREDICTIONS)
    validation_predictions = prediction_map(VALIDATION_PREDICTIONS)
    if set(dev_predictions) != {q.qid for q in dev}:
        raise ValueError("Development predictions do not match eligible questions")
    if set(validation_predictions) != {q.qid for q in validation}:
        raise ValueError("Validation predictions do not match eligible questions")

    records = {}
    for question in dev:
        index = indexes[question.set_id]
        features, pool = deep_feature_pool(index, question.text)
        records[question.qid] = (index, question.text, features, pool, set(question.gold))

    method_names = ("canonical_learned_cover", "deep_pairwise", "deep_set_aware")
    dev_outcomes = {name: [] for name in method_names}
    folds = []
    for fold in range(FOLDS):
        training = [q for q in dev if fold_of(q.set_id) != fold]
        held = [q for q in dev if fold_of(q.set_id) == fold]
        pair_records = [(records[q.qid][2], records[q.qid][3], records[q.qid][4])
                        for q in training]
        pairwise_weights, _ = train(pair_records)
        set_weights, set_metadata = train_set_aware(
            [records[q.qid] for q in training], pairwise_weights)
        counts = {name: 0 for name in method_names}
        for question in held:
            index, text, features, pool, gold = records[question.qid]
            prediction = dev_predictions[question.qid]
            if tuple(prediction["gold_indices"]) != question.gold:
                raise ValueError(f"Development gold changed: {question.qid}")
            rankings = {
                "canonical_learned_cover": prediction["rankings"]["learned_cover"],
                "deep_pairwise": select_topk(index, text, features, pool, pairwise_weights),
                "deep_set_aware": select_topk(index, text, features, pool, set_weights),
            }
            for name, ranking in rankings.items():
                dev_outcomes[name].append((question, ranking))
                counts[name] += gold <= set(ranking)
        folds.append({"fold": fold, "train_questions": len(training),
                      "held_questions": len(held), "held_complete_counts": counts,
                      "set_training": set_metadata,
                      "pairwise_weights": [round(value, 8) for value in pairwise_weights],
                      "set_aware_weights": [round(value, 8) for value in set_weights]})

    full_pair_records = [(records[q.qid][2], records[q.qid][3], records[q.qid][4]) for q in dev]
    final_pairwise_weights, _ = train(full_pair_records)
    final_set_weights, final_set_metadata = train_set_aware(
        [records[q.qid] for q in dev], final_pairwise_weights)
    validation_outcomes = {name: [] for name in method_names}
    for question in validation:
        prediction = validation_predictions[question.qid]
        if tuple(prediction["gold_indices"]) != question.gold:
            raise ValueError(f"Validation gold changed: {question.qid}")
        index = indexes[question.set_id]
        features, pool = deep_feature_pool(index, question.text)
        rankings = {
            "canonical_learned_cover": prediction["rankings"]["learned_cover"],
            "deep_pairwise": select_topk(
                index, question.text, features, pool, final_pairwise_weights),
            "deep_set_aware": select_topk(
                index, question.text, features, pool, final_set_weights),
        }
        for name, ranking in rankings.items():
            validation_outcomes[name].append((question, ranking))

    def comparisons(outcomes) -> dict:
        return {
            "set_aware_minus_canonical": paired_cluster_bootstrap(
                outcomes["canonical_learned_cover"], outcomes["deep_set_aware"]),
            "set_aware_minus_deep_pairwise": paired_cluster_bootstrap(
                outcomes["deep_pairwise"], outcomes["deep_set_aware"]),
        }

    def cells(outcomes) -> dict:
        return {
            "set_aware_vs_canonical": paired_cells(
                outcomes["canonical_learned_cover"], outcomes["deep_set_aware"]),
            "set_aware_vs_deep_pairwise": paired_cells(
                outcomes["deep_pairwise"], outcomes["deep_set_aware"]),
        }

    report = {
        "status": "structured_top5_dev_crossfit_validation_descriptive_no_holdout",
        "method": (
            "Pairwise deep-pool weights followed by four deterministic structured perceptron epochs; "
            "each update raises missing-gold mean features and lowers false-selected mean features."
        ),
        "pool_size_per_lexical_method": POOL_SIZE,
        "dev_crossfit": {
            "folds": folds,
            "metrics": {name: summarize(rows)["multihop"] for name, rows in dev_outcomes.items()},
            "comparisons": comparisons(dev_outcomes),
            "paired_cells": cells(dev_outcomes),
        },
        "final_training": {
            "set_training": final_set_metadata,
            "pairwise_weights": {name: round(value, 8)
                                 for name, value in zip(FEATURE_NAMES, final_pairwise_weights)},
            "set_aware_weights": {name: round(value, 8)
                                  for name, value in zip(FEATURE_NAMES, final_set_weights)},
        },
        "validation": {
            "metrics": {name: summarize(rows)["multihop"]
                        for name, rows in validation_outcomes.items()},
            "comparisons": comparisons(validation_outcomes),
            "paired_cells": cells(validation_outcomes),
        },
        "provenance": {
            "labels_sha256": sha256(label_path.read_bytes()).hexdigest(),
            "qa_sha256": sha256(qa_path.read_bytes()).hexdigest(),
            "dev_predictions_sha256": sha256(DEV_PREDICTIONS.read_bytes()).hexdigest(),
            "validation_predictions_sha256": sha256(VALIDATION_PREDICTIONS.read_bytes()).hexdigest(),
            "deep_pool_code_sha256": sha256(
                Path(__file__).with_name("deep_pool_ablation.py").read_bytes()).hexdigest(),
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "The structured update is a development ablation over incomplete upstream gold indices.",
            "Its fixed hyperparameters were not selected by validation, but validation has been inspected elsewhere.",
            "A perceptron update is order-sensitive; the order and seed are fixed and reported.",
            "The canonical selector and both sealed holdouts remain unchanged.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "dev": {name: row["all_gold@5"] for name, row in report["dev_crossfit"]["metrics"].items()},
        "validation": {name: row["all_gold@5"]
                       for name, row in report["validation"]["metrics"].items()},
    }))


if __name__ == "__main__":
    main()
