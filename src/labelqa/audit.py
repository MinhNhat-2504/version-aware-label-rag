"""Export model disagreements for human evidence adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_labels, load_questions, split_label, validate_data
from .retrieval import LabelIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/upstream/FDARxBench/data"))
    parser.add_argument("--split", choices=["dev", "validation"], default="validation")
    parser.add_argument("--candidate", choices=["section_cover", "clause_cover", "ontology_cover"], default="ontology_cover")
    parser.add_argument("--output", type=Path, default=Path("results/evidence_audit.jsonl"))
    parser.add_argument("--per-category", type=int, default=20)
    args = parser.parse_args()

    labels = load_labels(args.source / "labels" / "labels.jsonl")
    questions = load_questions(args.source / "qa" / "qa.jsonl")
    validate_data(labels, questions)
    indexes: dict[str, LabelIndex] = {}
    selected: dict[str, list[dict]] = {
        "candidate_gain": [], "candidate_loss": [], "both_miss": [], "large_gold_set": []
    }
    selected_labels: dict[str, set[str]] = {name: set() for name in selected}
    for q in questions:
        if q.task != "multihop" or split_label(q.set_id) != args.split:
            continue
        if q.set_id not in indexes:
            indexes[q.set_id] = LabelIndex(labels[q.set_id])
        index = indexes[q.set_id]
        baseline = index.retrieve(q.text, "bm25")
        candidate = index.retrieve(q.text, args.candidate)
        gold = set(q.gold)
        base_complete = gold <= set(baseline)
        candidate_complete = gold <= set(candidate)
        if len(gold) > 5:
            category = "large_gold_set"
        elif candidate_complete and not base_complete:
            category = "candidate_gain"
        elif base_complete and not candidate_complete:
            category = "candidate_loss"
        elif not base_complete and not candidate_complete:
            category = "both_miss"
        else:
            continue
        if len(selected[category]) >= args.per_category or q.set_id in selected_labels[category]:
            continue
        selected_labels[category].add(q.set_id)
        union = sorted(gold | set(baseline) | set(candidate))
        selected[category].append({
            "category": category,
            "qid": q.qid,
            "set_id": q.set_id,
            "drug_name": labels[q.set_id].drug_name,
            "question": q.text,
            "gold_indices": list(q.gold),
            "bm25_top5": baseline,
            "candidate_method": args.candidate,
            "candidate_top5": candidate,
            "passages": {str(i): labels[q.set_id].chunks[i] for i in union},
            "human_review": {
                "acceptable_support_sets": [],
                "gold_is_complete": None,
                "question_is_answerable": None,
                "notes": "",
            },
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for category in selected:
            for row in selected[category]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "counts": {
        category: len(rows) for category, rows in selected.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
