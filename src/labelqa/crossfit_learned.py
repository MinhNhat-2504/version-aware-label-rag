"""Group-crossfit fixed evidence-selector ablations within development labels."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from .data import load_labels, load_questions, split_label
from .evaluate import paired_cluster_bootstrap, summarize
from .learned_retrieval import feature_pool, retrieve, train
from .retrieval import LabelIndex


FOLDS = 4
ABLATIONS = {
    "full": frozenset(),
    "no_tfidf_features_pool_retained": frozenset(("tfidf_relative", "tfidf_reciprocal_rank")),
    "no_structure_features": frozenset(("title_term_coverage", "section_intent_match",
                                         "best_clause_relative")),
}
SOURCE = Path("data/upstream/FDARxBench/data")
ROOT = Path("results")


def fold_of(set_id: str) -> int:
    return int.from_bytes(sha256(("crossfit:" + set_id).encode()).digest()[:8], "big") % FOLDS


def main() -> None:
    labels = load_labels(SOURCE / "labels/labels.jsonl")
    questions = [q for q in load_questions(SOURCE / "qa/qa.jsonl")
                 if split_label(q.set_id) == "dev" and q.task == "multihop"
                 and 0 < len(q.gold) <= 5]
    indexes = {set_id: LabelIndex(labels[set_id]) for set_id in {q.set_id for q in questions}}
    records = {}
    for q in questions:
        features, pool = feature_pool(indexes[q.set_id], q.text)
        records[q.qid] = (features, pool, set(q.gold))
    outcomes = {method: [] for method in ("bm25", "ontology_cover", "learned_rank", "learned_cover",
                                           "no_tfidf_features_pool_retained", "no_structure_features")}
    fold_reports = []
    predictions = []
    for fold in range(FOLDS):
        train_questions = [q for q in questions if fold_of(q.set_id) != fold]
        held_questions = [q for q in questions if fold_of(q.set_id) == fold]
        weights_by_name = {}
        for name, disabled in ABLATIONS.items():
            weights_by_name[name], _ = train([records[q.qid] for q in train_questions], disabled)
        fold_counts = Counter()
        for q in held_questions:
            index = indexes[q.set_id]
            rankings = {
                "bm25": index.retrieve(q.text, "bm25"),
                "ontology_cover": index.retrieve(q.text, "ontology_cover"),
                "learned_rank": retrieve(index, q.text, weights_by_name["full"]),
                "learned_cover": retrieve(index, q.text, weights_by_name["full"], coverage=True),
                "no_tfidf_features_pool_retained": retrieve(
                    index, q.text, weights_by_name["no_tfidf_features_pool_retained"], coverage=True),
                "no_structure_features": retrieve(index, q.text,
                                                   weights_by_name["no_structure_features"], coverage=True),
            }
            for method, ranking in rankings.items():
                outcomes[method].append((q, ranking))
                fold_counts[method] += set(q.gold) <= set(ranking)
            predictions.append({"qid": q.qid, "set_id": q.set_id, "fold": fold,
                                "gold_indices": list(q.gold), "rankings": rankings})
        fold_reports.append({
            "fold": fold, "train_questions": len(train_questions), "held_questions": len(held_questions),
            "train_labels": len({q.set_id for q in train_questions}),
            "held_labels": len({q.set_id for q in held_questions}),
            "complete_counts": dict(fold_counts),
            "weights": {name: [round(value, 6) for value in weights]
                        for name, weights in weights_by_name.items()},
        })
    if len({q.qid for q, _ in outcomes["bm25"]}) != len(questions):
        raise ValueError("Crossfit did not score every dev question exactly once")
    if any(fold["held_labels"] == 0 for fold in fold_reports):
        raise ValueError("Empty held-label fold")
    ROOT.mkdir(parents=True, exist_ok=True)
    predictions_path = ROOT / "dev_learned_crossfit_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    report = {
        "status": "four_fold_set_id_grouped_development_crossfit_no_validation_used_for_ablation",
        "folds": FOLDS,
        "fold_reports": fold_reports,
        "metrics": {name: summarize(rows)["multihop"] for name, rows in outcomes.items()},
        "comparisons": {
            name + "_minus_bm25": paired_cluster_bootstrap(outcomes["bm25"], rows)
            for name, rows in outcomes.items() if name != "bm25"
        },
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256(predictions_path.read_bytes()).hexdigest(),
        "source_sha256": {
            "labels": sha256((SOURCE / "labels/labels.jsonl").read_bytes()).hexdigest(),
            "qa": sha256((SOURCE / "qa/qa.jsonl").read_bytes()).hexdigest(),
        },
        "limitations": [
            "Ablations retain the same BM25/TF-IDF/ontology candidate pool; they remove only named ranking features.",
            "Upstream context indices are an incomplete proxy for all acceptable evidence.",
            "This crossfit is restricted to dev labels and does not establish external clinical performance.",
        ],
    }
    output = ROOT / "learned_selector_dev_crossfit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "n": len(questions),
                      "rates": {name: row["all_gold@5"] for name, row in report["metrics"].items()},
                      "held_labels": [row["held_labels"] for row in fold_reports]}))


if __name__ == "__main__":
    main()
