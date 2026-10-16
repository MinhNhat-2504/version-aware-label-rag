"""Evaluate an explicit clause-coverage bonus without changing the frozen selector."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .crossfit_learned import FOLDS, fold_of
from .data import load_labels, load_questions, split_label
from .evaluate import paired_cluster_bootstrap, summarize
from .learned_retrieval import feature_pool, retrieve, score, train
from .retrieval import CLAUSE_BOUNDARY, LabelIndex, tokenize


SOURCE = Path("data/upstream/FDARxBench/data")
OUTPUT = Path("results/clause_coverage_ablation.json")
WEIGHT_GRID = (0.0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32)


def clause_retrieve(index: LabelIndex, question: str, weights: list[float],
                    clause_weight: float, k: int = 5) -> list[int]:
    """Apply a marginal per-clause bonus on top of the frozen learned selector."""
    if clause_weight < 0:
        raise ValueError("clause_weight must be non-negative")
    features, pool = feature_pool(index, question)
    query_terms = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query_terms) or 1.0
    clauses = [part for part in CLAUSE_BOUNDARY.split(question) if len(tokenize(part)) >= 2]
    clause_scores = [index.bm25_scores(clause) for clause in clauses[:3]] if len(clauses) > 1 else []
    clause_maxima = [max(values, default=0.0) or 1.0 for values in clause_scores]
    clause_covered = [0.0] * len(clause_scores)
    selected: list[int] = []
    covered: set[str] = set()
    titles: set[str] = set()
    remaining = pool.copy()
    while remaining and len(selected) < k:
        def value(candidate: int) -> float:
            new_terms = (query_terms & set(index.terms[candidate])) - covered
            novelty = sum(index.idf.get(term, 0.0) for term in new_terms) / denominator
            diversity = float(index.title_lines[candidate] not in titles)
            clause_gain = sum(
                max(0.0, values[candidate] / maximum - previous)
                for values, maximum, previous in zip(clause_scores, clause_maxima, clause_covered)
            ) / max(len(clause_scores), 1)
            return (score(weights, features[candidate]) + 0.12 * novelty + 0.04 * diversity
                    + clause_weight * clause_gain)

        best = max(remaining, key=lambda candidate: (value(candidate), -candidate))
        selected.append(best)
        covered.update(query_terms & set(index.terms[best]))
        titles.add(index.title_lines[best])
        clause_covered = [max(previous, values[best] / maximum)
                          for values, maximum, previous
                          in zip(clause_scores, clause_maxima, clause_covered)]
        remaining.remove(best)
    return selected


def select_weight(examples: list[tuple[LabelIndex, str, set[int]]], weights: list[float],
                  grid: tuple[float, ...] = WEIGHT_GRID) -> tuple[float, dict[str, int]]:
    """Maximize complete-gold count on supplied development examples; prefer simplicity on ties."""
    if not grid or any(value < 0 for value in grid):
        raise ValueError("Weight grid must contain non-negative values")
    counts = {
        str(value): sum(gold <= set(clause_retrieve(index, question, weights, value))
                        for index, question, gold in examples)
        for value in grid
    }
    selected = max(grid, key=lambda value: (counts[str(value)], -value))
    return selected, counts


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
    records = {}
    for question in dev:
        features, pool = feature_pool(indexes[question.set_id], question.text)
        records[question.qid] = (features, pool, set(question.gold))

    dev_outcomes = {"learned_cover": [], "clause_cover_ablation": []}
    folds = []
    for fold in range(FOLDS):
        training = [q for q in dev if fold_of(q.set_id) != fold]
        held = [q for q in dev if fold_of(q.set_id) == fold]
        weights, _ = train([records[q.qid] for q in training])
        selected_weight, grid_counts = select_weight(
            [(indexes[q.set_id], q.text, set(q.gold)) for q in training], weights)
        counts = Counter()
        for question in held:
            index = indexes[question.set_id]
            baseline = retrieve(index, question.text, weights, coverage=True)
            candidate = clause_retrieve(index, question.text, weights, selected_weight)
            dev_outcomes["learned_cover"].append((question, baseline))
            dev_outcomes["clause_cover_ablation"].append((question, candidate))
            counts["learned_cover"] += set(question.gold) <= set(baseline)
            counts["clause_cover_ablation"] += set(question.gold) <= set(candidate)
        folds.append({"fold": fold, "train_questions": len(training),
                      "held_questions": len(held), "selected_weight": selected_weight,
                      "training_complete_counts_by_weight": grid_counts,
                      "held_complete_counts": dict(counts)})

    final_weights, _ = train([records[q.qid] for q in dev])
    final_weight, final_grid_counts = select_weight(
        [(indexes[q.set_id], q.text, set(q.gold)) for q in dev], final_weights)
    validation_outcomes = {"learned_cover": [], "clause_cover_ablation": []}
    for question in validation:
        index = indexes[question.set_id]
        validation_outcomes["learned_cover"].append(
            (question, retrieve(index, question.text, final_weights, coverage=True)))
        validation_outcomes["clause_cover_ablation"].append(
            (question, clause_retrieve(index, question.text, final_weights, final_weight)))

    report = {
        "status": "nested_grouped_dev_ablation_validation_descriptive_not_holdout",
        "method": (
            "Marginal normalized BM25 gain for up to three question clauses, added to the frozen "
            "learned-cover objective. Coefficients are selected without held-fold or validation gold."
        ),
        "weight_grid": list(WEIGHT_GRID),
        "dev_crossfit": {
            "folds": folds,
            "metrics": {name: summarize(rows)["multihop"] for name, rows in dev_outcomes.items()},
            "comparison": paired_cluster_bootstrap(
                dev_outcomes["learned_cover"], dev_outcomes["clause_cover_ablation"]),
        },
        "full_dev_selection": {"selected_weight": final_weight,
                               "complete_counts_by_weight": final_grid_counts},
        "validation": {
            "metrics": {name: summarize(rows)["multihop"]
                        for name, rows in validation_outcomes.items()},
            "comparison": paired_cluster_bootstrap(
                validation_outcomes["learned_cover"],
                validation_outcomes["clause_cover_ablation"]),
        },
        "provenance": {
            "labels_sha256": sha256(label_path.read_bytes()).hexdigest(),
            "qa_sha256": sha256(qa_path.read_bytes()).hexdigest(),
            "canonical_selector_code_sha256": sha256(
                Path(__file__).with_name("learned_retrieval.py").read_bytes()).hexdigest(),
            "ablation_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "All outcomes use upstream gold context indices, not independently adjudicated evidence.",
            "Validation has been used elsewhere in method development and is descriptive here.",
            "This ablation does not modify the frozen learned selector or open either holdout.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT),
                      "dev": {name: row["all_gold@5"]
                              for name, row in report["dev_crossfit"]["metrics"].items()},
                      "selected_weight": final_weight,
                      "validation": {name: row["all_gold@5"]
                                     for name, row in report["validation"]["metrics"].items()}}))


if __name__ == "__main__":
    main()
