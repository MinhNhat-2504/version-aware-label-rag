"""Measure retrieval headroom on development or validation without opening test."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .data import load_labels, load_questions, split_label
from .retrieval import LabelIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "validation"), default="dev")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = Path("data/upstream/FDARxBench/data")
    labels = load_labels(source / "labels/labels.jsonl")
    questions = [q for q in load_questions(source / "qa/qa.jsonl")
                 if split_label(q.set_id) == args.split and q.task == "multihop"
                 and 0 < len(q.gold) <= 5]
    indexes = {}
    totals = Counter()
    by_gold_size = {size: Counter() for size in range(1, 6)}
    examples = []
    for q in questions:
        if q.set_id not in indexes:
            indexes[q.set_id] = LabelIndex(labels[q.set_id])
        index = indexes[q.set_id]
        bm25 = index.retrieve(q.text, "bm25", 5)
        ontology = index.retrieve(q.text, "ontology_cover", 5)
        clause = index.retrieve(q.text, "clause_cover", 5)
        evidence = index.retrieve(q.text, "evidence_set", 5)
        top20 = index.retrieve(q.text, "bm25", 20)
        top50 = index.retrieve(q.text, "bm25", 50)
        gold = set(q.gold)
        hits = {"bm25": gold <= set(bm25), "ontology": gold <= set(ontology),
                "clause": gold <= set(clause), "evidence": gold <= set(evidence),
                "union_bm25_ontology": gold <= set(bm25 + ontology),
                "union_all": gold <= set(bm25 + ontology + clause + evidence),
                "bm25_top20": gold <= set(top20),
                "bm25_top50": gold <= set(top50)}
        totals.update(name for name, hit in hits.items() if hit)
        by_gold_size[len(gold)]["n"] += 1
        by_gold_size[len(gold)].update(name for name, hit in hits.items() if hit)
        if not hits["ontology"] and hits["bm25"]:
            examples.append({"qid": q.qid, "set_id": q.set_id, "question": q.text,
                             "gold": q.gold, "bm25": bm25, "ontology": ontology,
                             "missing_ontology": sorted(gold - set(ontology))})
    result = {"split": args.split, "task": "multihop", "eligible": len(questions),
              "complete_counts": dict(totals),
              "by_gold_size": {size: dict(counts) for size, counts in by_gold_size.items()},
              "oracle_union_is_diagnostic_not_a_method": True,
              "ontology_lost_bm25_examples": examples[:30]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "ontology_lost_bm25_examples"}))


if __name__ == "__main__":
    main()
