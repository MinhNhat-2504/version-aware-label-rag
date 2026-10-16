"""Evaluate a learned five-passage selector over a deeper lexical candidate pool."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path

from .crossfit_learned import FOLDS, fold_of
from .data import load_labels, load_questions, read_jsonl, split_label
from .evaluate import paired_cluster_bootstrap, summarize
from .learned_retrieval import FEATURE_NAMES, score, train
from .retrieval import CLAUSE_BOUNDARY, SECTION_INTENTS, LabelIndex, tokenize


SOURCE = Path("data/upstream/FDARxBench/data")
OUTPUT = Path("results/deep_pool_ablation.json")
DEV_PREDICTIONS = Path("results/dev_learned_crossfit_predictions.jsonl")
VALIDATION_PREDICTIONS = Path("results/validation_learned_predictions.jsonl")
POOL_SIZE = 50


def deep_feature_pool(index: LabelIndex, question: str,
                      pool_size: int = POOL_SIZE) -> tuple[dict[int, tuple[float, ...]], list[int]]:
    """Build the canonical feature set over deeper BM25 and TF-IDF rankings."""
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    bm25 = index.bm25_scores(question)
    tfidf = index.tfidf_scores(question)
    bm25_rank = sorted(range(len(bm25)), key=lambda i: (-bm25[i], i))
    tfidf_rank = sorted(range(len(tfidf)), key=lambda i: (-tfidf[i], i))
    ontology = index.retrieve(question, "ontology_cover", 5)
    pool = list(dict.fromkeys(bm25_rank[:pool_size] + tfidf_rank[:pool_size] + ontology))
    bm25_position = {candidate: rank for rank, candidate in enumerate(bm25_rank, 1)}
    tfidf_position = {candidate: rank for rank, candidate in enumerate(tfidf_rank, 1)}
    max_bm25 = max(bm25, default=0.0) or 1.0
    max_tfidf = max(tfidf, default=0.0) or 1.0
    query_terms = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query_terms) or 1.0
    intents = [headings for trigger, headings in SECTION_INTENTS if trigger.search(question)]
    clauses = [part for part in CLAUSE_BOUNDARY.split(question) if len(tokenize(part)) >= 2]
    clause_scores = [index.bm25_scores(clause) for clause in clauses[:3]] if len(clauses) > 1 else []
    clause_maxima = [max(values, default=0.0) or 1.0 for values in clause_scores]
    longest = math.log1p(max(index.lengths, default=0)) or 1.0
    features = {}
    for candidate in pool:
        terms = set(index.terms[candidate])
        title_terms = set(index.titles[candidate])
        title = index.title_lines[candidate]
        features[candidate] = (
            bm25[candidate] / max_bm25,
            1 / bm25_position[candidate],
            tfidf[candidate] / max_tfidf,
            1 / tfidf_position[candidate],
            sum(index.idf.get(term, 0.0) for term in query_terms & terms) / denominator,
            sum(index.idf.get(term, 0.0) for term in query_terms & title_terms) / denominator,
            float(any(any(heading in title for heading in headings) for headings in intents)),
            max((values[candidate] / maximum
                 for values, maximum in zip(clause_scores, clause_maxima)), default=0.0),
            1.0 - math.log1p(index.lengths[candidate]) / longest,
        )
    return features, pool


def deep_retrieve(index: LabelIndex, question: str, weights: list[float],
                  k: int = 5, pool_size: int = POOL_SIZE) -> list[int]:
    """Apply the frozen coverage objective to a deep candidate pool."""
    features, pool = deep_feature_pool(index, question, pool_size)
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


def prediction_map(path: Path) -> dict[str, dict]:
    rows = {row["qid"]: row for row in read_jsonl(path)}
    if not rows:
        raise ValueError(f"No predictions: {path}")
    return rows


def paired_cells(baseline, candidate) -> dict[str, int]:
    cells = {"both_complete": 0, "baseline_only": 0, "deep_pool_only": 0, "neither": 0}
    for (base_question, base_ranking), (new_question, new_ranking) in zip(baseline, candidate):
        if base_question.qid != new_question.qid or base_question.gold != new_question.gold:
            raise ValueError("Paired outcomes are not aligned")
        gold = set(base_question.gold)
        base_complete = gold <= set(base_ranking)
        new_complete = gold <= set(new_ranking)
        key = ("both_complete" if base_complete and new_complete else
               "baseline_only" if base_complete else
               "deep_pool_only" if new_complete else "neither")
        cells[key] += 1
    return cells


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
        features, pool = deep_feature_pool(indexes[question.set_id], question.text)
        records[question.qid] = (features, pool, set(question.gold))

    dev_outcomes = {"canonical_learned_cover": [], "deep_pool_learned_cover": []}
    folds = []
    for fold in range(FOLDS):
        training = [q for q in dev if fold_of(q.set_id) != fold]
        held = [q for q in dev if fold_of(q.set_id) == fold]
        weights, metadata = train([records[q.qid] for q in training])
        deep_complete = 0
        deep_pool_complete = 0
        for question in held:
            prediction = dev_predictions[question.qid]
            if tuple(prediction["gold_indices"]) != question.gold:
                raise ValueError(f"Development gold changed: {question.qid}")
            baseline = prediction["rankings"]["learned_cover"]
            candidate = deep_retrieve(indexes[question.set_id], question.text, weights)
            dev_outcomes["canonical_learned_cover"].append((question, baseline))
            dev_outcomes["deep_pool_learned_cover"].append((question, candidate))
            deep_complete += set(question.gold) <= set(candidate)
            deep_pool_complete += set(question.gold) <= set(records[question.qid][1])
        folds.append({
            "fold": fold,
            "train_questions": len(training),
            "held_questions": len(held),
            "held_labels": len({q.set_id for q in held}),
            "deep_pool_complete": deep_pool_complete,
            "deep_top5_complete": deep_complete,
            "weights": {name: round(value, 8) for name, value in zip(FEATURE_NAMES, weights)},
            "training_questions_with_a_gold_in_pool": metadata["questions_with_a_gold_in_pool"],
        })

    final_weights, final_metadata = train([records[q.qid] for q in dev])
    validation_outcomes = {"canonical_learned_cover": [], "deep_pool_learned_cover": []}
    validation_pool_complete = 0
    validation_pool_sizes = []
    for question in validation:
        prediction = validation_predictions[question.qid]
        if tuple(prediction["gold_indices"]) != question.gold:
            raise ValueError(f"Validation gold changed: {question.qid}")
        features, pool = deep_feature_pool(indexes[question.set_id], question.text)
        del features
        validation_pool_complete += set(question.gold) <= set(pool)
        validation_pool_sizes.append(len(pool))
        validation_outcomes["canonical_learned_cover"].append(
            (question, prediction["rankings"]["learned_cover"]))
        validation_outcomes["deep_pool_learned_cover"].append(
            (question, deep_retrieve(indexes[question.set_id], question.text, final_weights)))

    report = {
        "status": "grouped_dev_crossfit_validation_descriptive_no_holdout",
        "pool_size_per_lexical_method": POOL_SIZE,
        "method": "BM25 top 50 plus TF-IDF top 50 and ontology candidates; same learned features and five-passage coverage objective.",
        "dev_crossfit": {
            "folds": folds,
            "metrics": {name: summarize(rows)["multihop"] for name, rows in dev_outcomes.items()},
            "comparison": paired_cluster_bootstrap(
                dev_outcomes["canonical_learned_cover"],
                dev_outcomes["deep_pool_learned_cover"]),
            "paired_cells": paired_cells(
                dev_outcomes["canonical_learned_cover"],
                dev_outcomes["deep_pool_learned_cover"]),
        },
        "final_training": {
            "questions": len(dev),
            "questions_with_a_gold_in_pool": final_metadata["questions_with_a_gold_in_pool"],
            "complete_gold_in_deep_pool": sum(
                set(q.gold) <= set(records[q.qid][1]) for q in dev),
            "weights": {name: round(value, 8) for name, value in zip(FEATURE_NAMES, final_weights)},
        },
        "validation": {
            "deep_pool_complete_gold_oracle": validation_pool_complete,
            "mean_deep_pool_size": round(sum(validation_pool_sizes) / len(validation_pool_sizes), 2),
            "metrics": {name: summarize(rows)["multihop"]
                        for name, rows in validation_outcomes.items()},
            "comparison": paired_cluster_bootstrap(
                validation_outcomes["canonical_learned_cover"],
                validation_outcomes["deep_pool_learned_cover"]),
            "paired_cells": paired_cells(
                validation_outcomes["canonical_learned_cover"],
                validation_outcomes["deep_pool_learned_cover"]),
        },
        "provenance": {
            "labels_sha256": sha256(label_path.read_bytes()).hexdigest(),
            "qa_sha256": sha256(qa_path.read_bytes()).hexdigest(),
            "dev_predictions_sha256": sha256(DEV_PREDICTIONS.read_bytes()).hexdigest(),
            "validation_predictions_sha256": sha256(VALIDATION_PREDICTIONS.read_bytes()).hexdigest(),
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Upstream gold context indices are an incomplete evidence proxy.",
            "Validation has already been inspected and is descriptive rather than confirmatory.",
            "The deep candidate pool increases ranking cost and may add hard distractors.",
            "The canonical selector and both sealed holdouts remain unchanged.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "dev": {name: row["all_gold@5"] for name, row in report["dev_crossfit"]["metrics"].items()},
        "validation_pool_complete": validation_pool_complete,
        "validation": {name: row["all_gold@5"]
                       for name, row in report["validation"]["metrics"].items()},
    }))


if __name__ == "__main__":
    main()
