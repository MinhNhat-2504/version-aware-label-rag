"""Train a dev-only evidence selector and evaluate validation without using test."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .data import load_labels, load_questions, split_label
from .evaluate import paired_cluster_bootstrap, summarize
from .learned_retrieval import FEATURE_NAMES, feature_pool, retrieve, train
from .retrieval import LabelIndex


SOURCE = Path("data/upstream/FDARxBench/data")
ROOT = Path("results")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def evaluate_questions(questions, indexes, weights):
    outcomes = {method: [] for method in ("bm25", "tfidf", "rrf_lexical", "ontology_cover",
                                           "learned_rank", "learned_cover")}
    timings = defaultdict(list)
    predictions = []
    for question in questions:
        index = indexes[question.set_id]
        rankings = {}
        for method in outcomes:
            start = perf_counter()
            ranking = (retrieve(index, question.text, weights, coverage=method == "learned_cover")
                       if method.startswith("learned_") else index.retrieve(question.text, method))
            timings[method].append((perf_counter() - start) * 1000)
            outcomes[method].append((question, ranking))
            rankings[method] = ranking
        predictions.append({
            "qid": question.qid, "set_id": question.set_id, "task": question.task,
            "gold_indices": list(question.gold), "rankings": rankings,
            "complete_by_method": {name: set(question.gold) <= set(ranking)
                                   for name, ranking in rankings.items()},
        })
    return outcomes, timings, predictions


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
    records = []
    complete_pool = 0
    for question in dev:
        features, pool = feature_pool(indexes[question.set_id], question.text)
        gold = set(question.gold)
        records.append((features, pool, gold))
        complete_pool += gold <= set(pool)
    start = perf_counter()
    weights, training = train(records)
    training_seconds = round(perf_counter() - start, 3)
    training["complete_gold_in_candidate_pool"] = complete_pool
    model = {
        "split_used_for_fitting": "dev", "task": "multihop", "features": list(FEATURE_NAMES),
        "weights": {name: round(weight, 8) for name, weight in zip(FEATURE_NAMES, weights)},
        "training": training,
        "source_sha256": {"labels": digest(label_path), "qa": digest(qa_path)},
        "code_sha256": digest(Path(__file__).with_name("learned_retrieval.py")),
        "note": "Upstream context-index supervision; unadjudicated alternate evidence may be labeled negative.",
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "learned_selector_model.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dev_outcomes, _, _ = evaluate_questions(dev, indexes, weights)
    validation_outcomes, timings, predictions = evaluate_questions(validation, indexes, weights)
    validation_pool_complete = sum(
        set(question.gold) <= set(feature_pool(indexes[question.set_id], question.text)[1])
        for question in validation)
    prediction_path = ROOT / "validation_learned_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    comparisons = {
        "learned_rank_minus_bm25": paired_cluster_bootstrap(validation_outcomes["bm25"],
                                                              validation_outcomes["learned_rank"]),
        "learned_cover_minus_bm25": paired_cluster_bootstrap(validation_outcomes["bm25"],
                                                               validation_outcomes["learned_cover"]),
        "learned_cover_minus_ontology": paired_cluster_bootstrap(validation_outcomes["ontology_cover"],
                                                                   validation_outcomes["learned_cover"]),
    }
    report = {
        "status": "dev_trained_validation_evaluated_upstream_gold_proxy_not_clinical_result",
        "model_path": str(model_path), "model_sha256": digest(model_path),
        "training_seconds": training_seconds,
        "validation_complete_gold_in_candidate_pool_oracle": validation_pool_complete,
        "predictions_path": str(prediction_path), "predictions_sha256": digest(prediction_path),
        "dev_in_sample": {name: summarize(rows)["multihop"] for name, rows in dev_outcomes.items()},
        "validation": {name: summarize(rows)["multihop"] for name, rows in validation_outcomes.items()},
        "comparisons": comparisons,
        "mean_retrieval_ms_per_question": {
            name: round(sum(values) / len(values), 3) for name, values in timings.items()},
        "limitations": [
            "Training uses only dev label IDs and upstream context indices; validation is used once for assessment.",
            "Upstream gold indices do not cover all semantically equivalent passages.",
            "Local timing excludes model generation and varies by hardware and process state.",
            "No held-out test or new-label holdout was evaluated.",
        ],
    }
    report_path = ROOT / "learned_selector_validation.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "dev": {
        name: result["all_gold@5"] for name, result in report["dev_in_sample"].items()},
        "validation": {name: result["all_gold@5"] for name, result in report["validation"].items()},
        "pool_gold_complete_dev": complete_pool, "dev_n": len(dev), "validation_n": len(validation)}))


if __name__ == "__main__":
    main()
