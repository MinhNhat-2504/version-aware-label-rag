"""Post hoc diagnostics of the exploratory neural baselines (why rerankers lose on all-gold@5).

Reads the archived predictions of neural_baselines.py; no model is rerun and nothing is tuned.
Reports, per split: a none/partial/complete decomposition of each method's top five, how often the
BM25 top 20 (the reranker's input) and the selector's pool contain every gold passage, the share of
passages longer than the 512-token encoder limit, CPU time per question for the lexical methods, and
all-gold@5 of the selector without its coverage step (learned scores only, a post hoc ablation).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from .data import load_labels, load_questions, read_jsonl
from .learned_retrieval import feature_pool, retrieve as learned_retrieve
from .neural_baselines import DENSE_MODEL, MAX_TOKENS, SOURCE
from .prepare_holdout_eval import model_weights
from .retrieval import LabelIndex

METHODS = ("bm25", "tfidf", "rrf_lexical", "ontology_cover", "learned_cover", "bge_m3", "hybrid_bm25_bge",
           "rerank_bm25_top20", "rerank_hybrid_top20")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=Path("results/neural_baselines_predictions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/neural_baselines_diagnostics.json"))
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)
    labels = load_labels(SOURCE / "labels/labels.jsonl")
    weights, _ = model_weights()
    texts = {q.qid: q.text for q in load_questions(SOURCE / "qa/qa.jsonl")}
    rows_by_split = defaultdict(list)
    for row in read_jsonl(args.predictions):
        rows_by_split[row["split"]].append(row)
    report = {"status": "exploratory_post_hoc_diagnostic_of_neural_baselines", "predictions": str(args.predictions),
              "splits": {}}
    for split, rows in rows_by_split.items():
        indexes = {set_id: LabelIndex(labels[set_id]) for set_id in sorted({r["set_id"] for r in rows})}
        lengths = [len(tokenizer(chunk, add_special_tokens=True)["input_ids"])
                   for index in indexes.values() for chunk in index.label.chunks]
        outcome = {m: {"none": 0, "partial": 0, "complete": 0} for m in METHODS}
        bm25_top20 = selector_pool = rank_only = 0
        timing = defaultdict(float)
        for r in rows:
            index, gold = indexes[r["set_id"]], set(r["gold"])
            question = texts[r["qid"]]
            for m in METHODS:
                top = r[m][:5]
                hits = len(gold & set(top))
                outcome[m]["complete" if hits == len(gold) else "partial" if hits else "none"] += 1
            bm25_top20 += gold <= set(index.retrieve(question, "bm25", 20))
            selector_pool += gold <= set(feature_pool(index, question)[1])
            rank_only += gold <= set(learned_retrieve(index, question, weights, k=5, coverage=False))
            started = perf_counter()
            index.retrieve(question, "bm25", 5)
            timing["bm25_ms"] += 1000 * (perf_counter() - started)
            started = perf_counter()
            learned_retrieve(index, question, weights, k=5, coverage=True)
            timing["learned_cover_ms"] += 1000 * (perf_counter() - started)
        n = len(rows)
        report["splits"][split] = {
            "n_questions": n, "n_labels": len(indexes), "n_passages": len(lengths),
            "passage_tokens": {"median": median(lengths), "share_over_limit": round(sum(x > MAX_TOKENS for x in lengths) / len(lengths), 4)},
            "top5_outcome": outcome,
            "all_gold_in_bm25_top20": round(bm25_top20 / n, 4),
            "all_gold_in_selector_pool": round(selector_pool / n, 4),
            "learned_rank_without_coverage_all_gold@5": round(rank_only / n, 4),
            "cpu_ms_per_question": {key: round(value / n, 2) for key, value in timing.items()},
        }
        print(json.dumps({split: report["splits"][split]}), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
