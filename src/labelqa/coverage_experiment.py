"""Select a lexical coverage reranker on development labels only."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

from .data import load_labels, load_questions, split_label
from .retrieval import LabelIndex


NOVELTY = (0.0, 0.25, 0.5, 1.0, 2.0)
TITLE = (0.0, 0.15, 0.3)
DIVERSITY = (0.0, 0.1, 0.3)


def main() -> None:
    source = Path("data/upstream/FDARxBench/data")
    labels = load_labels(source / "labels/labels.jsonl")
    questions = [q for q in load_questions(source / "qa/qa.jsonl")
                 if split_label(q.set_id) == "dev" and q.task == "multihop"
                 and 0 < len(q.gold) <= 5]
    parameters = list(itertools.product(NOVELTY, TITLE, DIVERSITY))
    counts = Counter()
    baseline = 0
    indexes = {}
    for q in questions:
        if q.set_id not in indexes:
            indexes[q.set_id] = LabelIndex(labels[q.set_id])
        index = indexes[q.set_id]
        scores = index.bm25_scores(q.text)
        ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        gold = set(q.gold)
        baseline += gold <= set(ranking[:5])
        for novelty, title, diversity in parameters:
            selected = index.greedy_cover(q.text, ranking, 5, novelty, title, diversity, scores)
            counts[(novelty, title, diversity)] += gold <= set(selected)
    results = [{"novelty_weight": n, "title_weight": t, "diversity_weight": d,
                "complete": counts[n, t, d], "rate": round(counts[n, t, d] / len(questions), 4)}
               for n, t, d in parameters]
    results.sort(key=lambda r: (-r["complete"], r["novelty_weight"],
                                r["title_weight"], r["diversity_weight"]))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "selection_split": "dev", "task": "multihop", "n": len(questions),
              "baseline_bm25_complete": baseline,
              "criterion": "Highest dev all-gold@5 count; tie-break by smaller weights in novelty/title/diversity order.",
              "grid": {"novelty": NOVELTY, "title": TITLE, "diversity": DIVERSITY},
              "best": results[0], "all_results": results,
              "interpretation": "Upstream context-index diagnostic; alternate evidence remains unadjudicated."}
    output = Path("results/coverage_grid_dev.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(questions), "bm25_complete": baseline,
                      "best": results[0], "top_five": results[:5], "output": str(output)}))


if __name__ == "__main__":
    main()
