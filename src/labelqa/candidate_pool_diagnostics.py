"""Diagnose candidate-generation and top-five selection failures without changing retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .data import load_labels, load_questions, read_jsonl, split_label
from .learned_retrieval import feature_pool
from .retrieval import CLAUSE_BOUNDARY, LabelIndex, tokenize


SOURCE = Path("data/upstream/FDARxBench/data")
OUTPUT = Path("results/candidate_pool_diagnostics.json")
PREDICTIONS = {
    "dev": Path("results/dev_learned_crossfit_predictions.jsonl"),
    "validation": Path("results/validation_learned_predictions.jsonl"),
}
STRUCTURAL_METHODS = ("section_cover", "clause_cover", "evidence_set", "greedy_cover")


def failure_category(gold: set[int], pool: set[int], ranking: list[int]) -> str:
    if gold <= set(ranking):
        return "complete_top5"
    if gold <= pool:
        return "selection_miss_gold_available_in_pool"
    return "candidate_pool_miss"


def passage_count_bin(count: int) -> str:
    if count <= 25:
        return "01-25"
    if count <= 50:
        return "26-50"
    if count <= 100:
        return "51-100"
    return "101+"


def has_multiple_clauses(question: str) -> bool:
    return len([part for part in CLAUSE_BOUNDARY.split(question)
                if len(tokenize(part)) >= 2]) > 1


def analyze_split(split: str, questions, labels) -> dict:
    eligible = {q.qid: q for q in questions
                if split_label(q.set_id) == split and q.task == "multihop"
                and 0 < len(q.gold) <= 5}
    prediction_path = PREDICTIONS[split]
    predictions = {row["qid"]: row for row in read_jsonl(prediction_path)}
    if predictions.keys() != eligible.keys():
        missing = sorted(eligible.keys() - predictions.keys())[:5]
        extra = sorted(predictions.keys() - eligible.keys())[:5]
        raise ValueError(f"Prediction/question mismatch for {split}; missing={missing}, extra={extra}")
    indexes = {set_id: LabelIndex(labels[set_id])
               for set_id in {question.set_id for question in eligible.values()}}
    taxonomy = Counter()
    by_passage_count = defaultdict(Counter)
    by_clause_shape = defaultdict(Counter)
    variants = ("canonical",) + tuple(f"canonical_plus_{name}" for name in STRUCTURAL_METHODS) + (
        "canonical_plus_all_structural", "bm25_top50")
    variant_complete = Counter()
    variant_pool_sizes = Counter()
    candidate_miss_examples = []

    for qid, question in eligible.items():
        prediction = predictions[qid]
        if prediction["set_id"] != question.set_id or tuple(prediction["gold_indices"]) != question.gold:
            raise ValueError(f"Prediction provenance mismatch: {qid}")
        ranking = prediction["rankings"]["learned_cover"]
        index = indexes[question.set_id]
        canonical = set(feature_pool(index, question.text)[1])
        additions = {method: set(index.retrieve(question.text, method, k=5))
                     for method in STRUCTURAL_METHODS}
        pools = {"canonical": canonical}
        pools.update({f"canonical_plus_{method}": canonical | values
                      for method, values in additions.items()})
        pools["canonical_plus_all_structural"] = canonical.union(*additions.values())
        pools["bm25_top50"] = set(index.retrieve(question.text, "bm25", k=50))
        gold = set(question.gold)
        category = failure_category(gold, canonical, ranking)
        taxonomy[category] += 1
        by_passage_count[passage_count_bin(len(index.label.chunks))][category] += 1
        clause_shape = "multiple_detected_clauses" if has_multiple_clauses(question.text) else "single_or_unsplit"
        by_clause_shape[clause_shape][category] += 1
        for name in variants:
            variant_complete[name] += gold <= pools[name]
            variant_pool_sizes[name] += len(pools[name])
        if category == "candidate_pool_miss" and len(candidate_miss_examples) < 12:
            candidate_miss_examples.append({
                "qid": qid,
                "set_id": question.set_id,
                "question": question.text,
                "label_passages": len(index.label.chunks),
                "gold_indices": list(question.gold),
                "missing_from_canonical": sorted(gold - canonical),
                "recovered_by": [name for name in variants[1:] if gold <= pools[name]],
            })

    n = len(eligible)
    return {
        "questions": n,
        "labels": len({q.set_id for q in eligible.values()}),
        "taxonomy": dict(taxonomy),
        "candidate_pool_complete": variant_complete["canonical"],
        "selection_complete": taxonomy["complete_top5"],
        "selection_success_given_complete_pool": round(
            taxonomy["complete_top5"] / variant_complete["canonical"], 4),
        "candidate_variants": {
            name: {
                "complete_gold_count": variant_complete[name],
                "complete_gold_rate": round(variant_complete[name] / n, 4),
                "recovered_over_canonical": variant_complete[name] - variant_complete["canonical"],
                "mean_pool_size": round(variant_pool_sizes[name] / n, 2),
            }
            for name in variants
        },
        "by_label_passage_count": {name: dict(counts)
                                   for name, counts in sorted(by_passage_count.items())},
        "by_question_clause_shape": {name: dict(counts)
                                     for name, counts in sorted(by_clause_shape.items())},
        "candidate_pool_miss_examples": candidate_miss_examples,
        "predictions_sha256": sha256(prediction_path.read_bytes()).hexdigest(),
    }


def main() -> None:
    label_path = SOURCE / "labels/labels.jsonl"
    qa_path = SOURCE / "qa/qa.jsonl"
    labels = load_labels(label_path)
    questions = load_questions(qa_path)
    report = {
        "status": "development_diagnostic_upstream_gold_proxy_no_holdout",
        "splits": {split: analyze_split(split, questions, labels) for split in PREDICTIONS},
        "provenance": {
            "labels_sha256": sha256(label_path.read_bytes()).hexdigest(),
            "qa_sha256": sha256(qa_path.read_bytes()).hexdigest(),
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Candidate completeness uses upstream context indices, which omit unannotated equivalent evidence.",
            "Structural additions are oracle pool diagnostics, not deployable top-five retrieval methods.",
            "Validation is descriptive because it has already been inspected during method development.",
            "Neither the benchmark test split nor the temporal holdout is accessed.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "splits": {name: {"taxonomy": row["taxonomy"],
                           "candidate_variants": row["candidate_variants"]}
                   for name, row in report["splits"].items()},
    }))


if __name__ == "__main__":
    main()
