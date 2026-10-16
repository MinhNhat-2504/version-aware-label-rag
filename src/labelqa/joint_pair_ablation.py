"""Evaluate a learned joint evidence-pair selector over the deep lexical pool."""

from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import random

from .crossfit_learned import FOLDS, fold_of
from .data import load_labels, load_questions, split_label
from .deep_pool_ablation import (
    DEV_PREDICTIONS,
    VALIDATION_PREDICTIONS,
    deep_feature_pool,
    paired_cells,
    prediction_map,
)
from .evaluate import paired_cluster_bootstrap, summarize
from .learned_retrieval import score, train
from .retrieval import CLAUSE_BOUNDARY, LabelIndex, tokenize


SOURCE = Path("data/upstream/FDARxBench/data")
OUTPUT = Path("results/joint_pair_ablation.json")
PAIR_FEATURE_NAMES = (
    "mean_passage_relevance",
    "minimum_passage_relevance",
    "union_query_coverage",
    "coverage_complementarity",
    "mean_query_coverage",
    "minimum_query_coverage",
    "title_diversity",
    "any_section_intent_match",
    "both_section_intent_match",
    "two_clause_assignment_coverage",
)
PAIR_EPOCHS = 8
PAIR_LEARNING_RATE = 0.03
PAIR_SEED = 20261017
ONE_GOLD_NEGATIVES = 12
ZERO_GOLD_NEGATIVES = 8


def normalized_passage_scores(features: dict[int, tuple[float, ...]], pool: list[int],
                              weights: list[float]) -> dict[int, float]:
    """Normalize learned passage scores within a question to [0, 1]."""
    values = {candidate: score(weights, features[candidate]) for candidate in pool}
    low = min(values.values(), default=0.0)
    high = max(values.values(), default=0.0)
    span = high - low
    if span <= 1e-12:
        return {candidate: 0.0 for candidate in pool}
    return {candidate: (value - low) / span for candidate, value in values.items()}


def pair_features(index: LabelIndex, question: str,
                  features: dict[int, tuple[float, ...]], relevance: dict[int, float],
                  first: int, second: int) -> tuple[float, ...]:
    """Construct query-only interaction features for one unordered passage pair."""
    query_terms = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query_terms) or 1.0
    first_terms = set(index.terms[first])
    second_terms = set(index.terms[second])
    union_coverage = sum(
        index.idf.get(term, 0.0)
        for term in query_terms & (first_terms | second_terms)
    ) / denominator
    individual_coverage = (features[first][4], features[second][4])

    clauses = [part for part in CLAUSE_BOUNDARY.split(question) if len(tokenize(part)) >= 2]
    assignment = 0.0
    if len(clauses) >= 2:
        clause_terms = [set(tokenize(part)) for part in clauses[:2]]

        def coverage(candidate_terms: set[str], terms: set[str]) -> float:
            clause_denominator = sum(index.idf.get(term, 0.0) for term in terms) or 1.0
            return sum(index.idf.get(term, 0.0)
                       for term in terms & candidate_terms) / clause_denominator

        direct = (coverage(first_terms, clause_terms[0])
                  + coverage(second_terms, clause_terms[1])) / 2
        swapped = (coverage(second_terms, clause_terms[0])
                   + coverage(first_terms, clause_terms[1])) / 2
        assignment = max(direct, swapped)

    passage_relevance = (relevance[first], relevance[second])
    intents = (features[first][6], features[second][6])
    return (
        sum(passage_relevance) / 2,
        min(passage_relevance),
        union_coverage,
        max(0.0, union_coverage - max(individual_coverage)),
        sum(individual_coverage) / 2,
        min(individual_coverage),
        float(index.title_lines[first] != index.title_lines[second]),
        float(any(intents)),
        float(all(intents)),
        assignment,
    )


def pair_score(weights: list[float], values: tuple[float, ...]) -> float:
    return sum(weight * value for weight, value in zip(weights, values))


def training_differences(records, passage_weights: list[float]) -> tuple[list[list[float]], dict]:
    """Create positive-pair minus hard-negative feature differences."""
    differences: list[list[float]] = []
    eligible_questions = 0
    one_gold_count = 0
    zero_gold_count = 0
    for index, question, features, pool, gold in records:
        if len(gold) != 2 or not gold <= set(pool):
            continue
        eligible_questions += 1
        relevance = normalized_passage_scores(features, pool, passage_weights)
        positive_pair = tuple(sorted(gold))
        positive = pair_features(index, question, features, relevance, *positive_pair)

        def heuristic(pair: tuple[int, int]) -> tuple[float, tuple[int, int]]:
            values = pair_features(index, question, features, relevance, *pair)
            value = values[0] + 0.5 * values[2] + 0.25 * values[3] + 0.25 * values[9]
            return value, tuple(-candidate for candidate in pair)

        one_gold = [pair for pair in combinations(pool, 2)
                    if len(set(pair) & gold) == 1]
        one_gold.sort(key=heuristic, reverse=True)
        ranked_passages = sorted(pool, key=lambda candidate: (-relevance[candidate], candidate))[:12]
        zero_gold = [pair for pair in combinations(ranked_passages, 2)
                     if not (set(pair) & gold)]
        zero_gold.sort(key=heuristic, reverse=True)
        negatives = (one_gold[:ONE_GOLD_NEGATIVES]
                     + zero_gold[:ZERO_GOLD_NEGATIVES])
        one_gold_count += min(len(one_gold), ONE_GOLD_NEGATIVES)
        zero_gold_count += min(len(zero_gold), ZERO_GOLD_NEGATIVES)
        for negative_pair in negatives:
            negative = pair_features(index, question, features, relevance, *negative_pair)
            differences.append([a - b for a, b in zip(positive, negative)])
    return differences, {
        "eligible_complete_pair_questions": eligible_questions,
        "one_gold_negative_pairs": one_gold_count,
        "zero_gold_negative_pairs": zero_gold_count,
        "training_pair_comparisons": len(differences),
    }


def train_pair_model(records, passage_weights: list[float]) -> tuple[list[float], dict]:
    """Train a deterministic pairwise logistic ranker for the complete gold pair."""
    differences, metadata = training_differences(records, passage_weights)
    weights = [0.0] * len(PAIR_FEATURE_NAMES)
    weights[0] = 1.0
    weights[2] = 0.5
    generator = random.Random(PAIR_SEED)
    order = list(range(len(differences)))
    for epoch in range(PAIR_EPOCHS):
        generator.shuffle(order)
        rate = PAIR_LEARNING_RATE / (1 + 0.2 * epoch)
        for position in order:
            difference = differences[position]
            margin = max(-40.0, min(40.0, pair_score(weights, tuple(difference))))
            gradient = 1.0 / (1.0 + math.exp(margin))
            for column, value in enumerate(difference):
                weights[column] += rate * (gradient * value - 0.0001 * weights[column])
    metadata.update({
        "epochs": PAIR_EPOCHS,
        "learning_rate": PAIR_LEARNING_RATE,
        "seed": PAIR_SEED,
        "features": list(PAIR_FEATURE_NAMES),
    })
    return weights, metadata


def joint_pair_retrieve(index: LabelIndex, question: str,
                        features: dict[int, tuple[float, ...]], pool: list[int],
                        passage_weights: list[float], pair_weights: list[float],
                        k: int = 5) -> tuple[list[int], tuple[int, int]]:
    """Select the best evidence pair, then fill the remaining passage budget."""
    if k < 2:
        raise ValueError("joint pair retrieval requires k >= 2")
    if len(pool) < 2:
        ranking = sorted(pool, key=lambda candidate: (-score(
            passage_weights, features[candidate]), candidate))[:k]
        return ranking, tuple(ranking)  # type: ignore[return-value]
    relevance = normalized_passage_scores(features, pool, passage_weights)

    def value(pair: tuple[int, int]) -> float:
        return pair_score(pair_weights, pair_features(
            index, question, features, relevance, *pair))

    best_pair = max(combinations(pool, 2), key=lambda pair: (value(pair), -pair[0], -pair[1]))
    selected = list(best_pair)
    query_terms = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query_terms) or 1.0
    covered = query_terms & (set(index.terms[best_pair[0]]) | set(index.terms[best_pair[1]]))
    titles = {index.title_lines[candidate] for candidate in selected}
    remaining = [candidate for candidate in pool if candidate not in selected]
    while remaining and len(selected) < k:
        def fill_value(candidate: int) -> float:
            new_terms = (query_terms & set(index.terms[candidate])) - covered
            novelty = sum(index.idf.get(term, 0.0) for term in new_terms) / denominator
            diversity = float(index.title_lines[candidate] not in titles)
            return score(passage_weights, features[candidate]) + 0.12 * novelty + 0.04 * diversity

        candidate = max(remaining, key=lambda item: (fill_value(item), -item))
        selected.append(candidate)
        covered.update(query_terms & set(index.terms[candidate]))
        titles.add(index.title_lines[candidate])
        remaining.remove(candidate)
    return selected, best_pair


def main() -> None:
    label_path = SOURCE / "labels/labels.jsonl"
    qa_path = SOURCE / "qa/qa.jsonl"
    labels = load_labels(label_path)
    questions = load_questions(qa_path)
    dev = [q for q in questions if split_label(q.set_id) == "dev"
           and q.task == "multihop" and len(q.gold) == 2]
    validation = [q for q in questions if split_label(q.set_id) == "validation"
                  and q.task == "multihop" and len(q.gold) == 2]
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

    names = ("canonical_learned_cover", "deep_independent_top5", "joint_pair")
    dev_outcomes = {name: [] for name in names}
    folds = []
    for fold in range(FOLDS):
        training = [q for q in dev if fold_of(q.set_id) != fold]
        held = [q for q in dev if fold_of(q.set_id) == fold]
        passage_weights, _ = train([
            (records[q.qid][2], records[q.qid][3], records[q.qid][4]) for q in training
        ])
        pair_weights, pair_metadata = train_pair_model(
            [records[q.qid] for q in training], passage_weights)
        counts = {name: 0 for name in names}
        exact_pairs = 0
        for question in held:
            index, text, features, pool, gold = records[question.qid]
            prediction = dev_predictions[question.qid]
            if tuple(prediction["gold_indices"]) != question.gold:
                raise ValueError(f"Development gold changed: {question.qid}")
            deep_ranking = sorted(
                pool, key=lambda candidate: (-score(passage_weights, features[candidate]), candidate)
            )[:5]
            joint_ranking, selected_pair = joint_pair_retrieve(
                index, text, features, pool, passage_weights, pair_weights)
            rankings = {
                "canonical_learned_cover": prediction["rankings"]["learned_cover"],
                "deep_independent_top5": deep_ranking,
                "joint_pair": joint_ranking,
            }
            exact_pairs += gold == set(selected_pair)
            for name, ranking in rankings.items():
                dev_outcomes[name].append((question, ranking))
                counts[name] += gold <= set(ranking)
        folds.append({
            "fold": fold,
            "train_questions": len(training),
            "held_questions": len(held),
            "held_complete_counts": counts,
            "held_exact_pair_at_1": exact_pairs,
            "pair_training": pair_metadata,
            "pair_weights": {name: round(value, 8)
                             for name, value in zip(PAIR_FEATURE_NAMES, pair_weights)},
        })

    full_passage_weights, _ = train([
        (records[q.qid][2], records[q.qid][3], records[q.qid][4]) for q in dev
    ])
    full_pair_weights, full_pair_metadata = train_pair_model(
        [records[q.qid] for q in dev], full_passage_weights)
    validation_outcomes = {name: [] for name in names}
    validation_exact_pairs = 0
    validation_pool_complete = 0
    for question in validation:
        prediction = validation_predictions[question.qid]
        if tuple(prediction["gold_indices"]) != question.gold:
            raise ValueError(f"Validation gold changed: {question.qid}")
        index = indexes[question.set_id]
        features, pool = deep_feature_pool(index, question.text)
        gold = set(question.gold)
        validation_pool_complete += gold <= set(pool)
        deep_ranking = sorted(
            pool, key=lambda candidate: (-score(full_passage_weights, features[candidate]), candidate)
        )[:5]
        joint_ranking, selected_pair = joint_pair_retrieve(
            index, question.text, features, pool, full_passage_weights, full_pair_weights)
        validation_exact_pairs += gold == set(selected_pair)
        rankings = {
            "canonical_learned_cover": prediction["rankings"]["learned_cover"],
            "deep_independent_top5": deep_ranking,
            "joint_pair": joint_ranking,
        }
        for name, ranking in rankings.items():
            validation_outcomes[name].append((question, ranking))

    def comparison(outcomes, baseline: str) -> dict:
        return {
            "bootstrap": paired_cluster_bootstrap(outcomes[baseline], outcomes["joint_pair"]),
            "paired_cells": paired_cells(outcomes[baseline], outcomes["joint_pair"]),
        }

    report = {
        "status": "joint_pair_dev_crossfit_validation_descriptive_no_holdout",
        "method": (
            "A deep-pool passage ranker supplies relevance features. A second logistic ranker "
            "scores unordered passage pairs using relevance, union coverage, complementarity, "
            "section diversity and two-clause assignment; the best pair is selected before the "
            "remaining three passages are filled by the frozen coverage objective."
        ),
        "dev_crossfit": {
            "folds": folds,
            "exact_pair_at_1": sum(fold["held_exact_pair_at_1"] for fold in folds),
            "metrics": {name: summarize(rows)["multihop"]
                        for name, rows in dev_outcomes.items()},
            "joint_minus_canonical": comparison(dev_outcomes, "canonical_learned_cover"),
            "joint_minus_deep_independent": comparison(
                dev_outcomes, "deep_independent_top5"),
        },
        "final_training": {
            "pair_training": full_pair_metadata,
            "passage_weights": [round(value, 8) for value in full_passage_weights],
            "pair_weights": {name: round(value, 8)
                             for name, value in zip(PAIR_FEATURE_NAMES, full_pair_weights)},
        },
        "validation": {
            "deep_pool_complete_gold_oracle": validation_pool_complete,
            "exact_pair_at_1": validation_exact_pairs,
            "metrics": {name: summarize(rows)["multihop"]
                        for name, rows in validation_outcomes.items()},
            "joint_minus_canonical": comparison(validation_outcomes, "canonical_learned_cover"),
            "joint_minus_deep_independent": comparison(
                validation_outcomes, "deep_independent_top5"),
        },
        "provenance": {
            "labels_sha256": sha256(label_path.read_bytes()).hexdigest(),
            "qa_sha256": sha256(qa_path.read_bytes()).hexdigest(),
            "dev_predictions_sha256": sha256(DEV_PREDICTIONS.read_bytes()).hexdigest(),
            "validation_predictions_sha256": sha256(VALIDATION_PREDICTIONS.read_bytes()).hexdigest(),
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "The benchmark endpoint has exactly two upstream gold passages per question; the pair objective is specialized to that proxy.",
            "Upstream gold indices may omit equivalent passages.",
            "Validation has already been inspected and is descriptive rather than confirmatory.",
            "The canonical selector and both sealed holdouts remain unchanged.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "dev": {name: row["all_gold@5"] for name, row in report["dev_crossfit"]["metrics"].items()},
        "dev_exact_pair_at_1": report["dev_crossfit"]["exact_pair_at_1"],
        "validation": {name: row["all_gold@5"]
                       for name, row in report["validation"]["metrics"].items()},
        "validation_exact_pair_at_1": validation_exact_pairs,
    }))


if __name__ == "__main__":
    main()
