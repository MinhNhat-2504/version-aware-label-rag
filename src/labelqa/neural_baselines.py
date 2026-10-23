"""Exploratory neural retrieval baselines on FDARxBench validation and test (post hoc).

Plan: paper/EXPLORATORY_PLAN_NEURAL_BASELINES.md, written before these runs. Adds BGE-M3
dense retrieval, BM25+BGE-M3 reciprocal rank fusion and a BGE-reranker-v2-m3 cross-encoder
over the BM25 (or hybrid) top 20, plus budget sensitivity (k = 3, 5, 10) and a breakdown by
question source. The preregistered test result is unchanged; everything here is exploratory.
Requires torch and transformers (optional dependencies, not used by the core pipeline).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

from .data import load_labels, load_questions, read_jsonl, split_label
from .dense_baseline import rank_rrf
from .learned_retrieval import retrieve as learned_retrieve
from .prepare_holdout_eval import model_weights
from .retrieval import LabelIndex

SOURCE = Path("data/upstream/FDARxBench/data")
DENSE_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
MAX_TOKENS = 512
RERANK_DEPTH = 20
KS = (3, 5, 10)
SEED = 20261024
DRAWS = 10_000


def eligible(questions, split: str):
    return [q for q in questions if split_label(q.set_id) == split and q.task == "multihop" and 0 < len(q.gold) <= 5]


class Encoders:
    """Half-precision GPU encoders with recorded model revisions."""

    def __init__(self, device: str):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, device
        self.dense_tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL)
        self.dense = AutoModel.from_pretrained(DENSE_MODEL, torch_dtype=torch.float16).to(device).eval()
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            RERANK_MODEL, torch_dtype=torch.float16).to(device).eval()
        self.revisions = {DENSE_MODEL: getattr(self.dense.config, "_commit_hash", None),
                          RERANK_MODEL: getattr(self.reranker.config, "_commit_hash", None)}

    def embed(self, texts: list[str], batch: int = 32):
        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch):
                encoded = self.dense_tokenizer(texts[start:start + batch], padding=True, truncation=True,
                                               max_length=MAX_TOKENS, return_tensors="pt").to(self.device)
                cls = self.dense(**encoded).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls.float(), dim=-1).cpu())
        return torch.cat(out)

    def rerank(self, query: str, passages: list[str], batch: int = 20) -> list[float]:
        torch = self.torch
        scores = []
        with torch.inference_mode():
            for start in range(0, len(passages), batch):
                pairs = [[query, passage] for passage in passages[start:start + batch]]
                encoded = self.rerank_tokenizer(pairs, padding=True, truncation=True, max_length=MAX_TOKENS,
                                                return_tensors="pt").to(self.device)
                scores += self.reranker(**encoded).logits.view(-1).float().cpu().tolist()
        return scores


def rerank_top(encoders: Encoders, query: str, chunks: tuple[str, ...], ranking: list[int]) -> list[int]:
    head = ranking[:RERANK_DEPTH]
    scores = encoders.rerank(query, [chunks[i] for i in head])
    reordered = [i for _, i in sorted(zip(scores, head), key=lambda pair: (-pair[0], pair[1]))]
    return reordered + [i for i in ranking if i not in set(head)]


def metrics(rows: list[dict], method: str) -> dict:
    result = {"n": len(rows)}
    for k in KS:
        result[f"all_gold@{k}"] = round(sum(set(r["gold"]) <= set(r[method][:k]) for r in rows) / len(rows), 4)
    result["recall@5"] = round(sum(len(set(r["gold"]) & set(r[method][:5])) / len(r["gold"]) for r in rows) / len(rows), 4)
    result["mrr@5"] = round(sum(next((1 / rank for rank, i in enumerate(r[method][:5], 1) if i in set(r["gold"])), 0.0)
                                for r in rows) / len(rows), 4)
    return result


def paired(rows: list[dict], candidate: str, baseline: str, k: int = 5) -> dict:
    by_label = defaultdict(list)
    for r in rows:
        gold = set(r["gold"])
        by_label[r["set_id"]].append((float(gold <= set(r[baseline][:k])), float(gold <= set(r[candidate][:k]))))
    labels = sorted(by_label)
    observations = [pair for label in labels for pair in by_label[label]]
    cells = {"both": sum(a and b for a, b in observations), "candidate_only": sum(b and not a for a, b in observations),
             "baseline_only": sum(a and not b for a, b in observations), "neither": sum(not a and not b for a, b in observations)}
    delta = sum(b - a for a, b in observations) / len(observations)
    generator = random.Random(SEED)
    samples = []
    for _ in range(DRAWS):
        chosen = generator.choices(labels, k=len(labels))
        samples.append(sum(b - a for label in chosen for a, b in by_label[label]) /
                       sum(len(by_label[label]) for label in chosen))
    samples.sort()
    return {"candidate": candidate, "baseline": baseline, "k": k, "n_questions": len(observations),
            "n_labels": len(labels), "difference_pp": round(100 * delta, 2),
            "ci95_pp": [round(100 * samples[int(0.025 * DRAWS)], 2), round(100 * samples[int(0.975 * DRAWS) - 1], 2)],
            "paired_cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--output", type=Path, default=Path("results/neural_baselines.json"))
    parser.add_argument("--predictions", type=Path, default=Path("results/neural_baselines_predictions.jsonl"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists() or args.predictions.exists():
        parser.error("Refusing to overwrite an exploratory report")
    labels = load_labels(SOURCE / "labels/labels.jsonl")
    questions = load_questions(SOURCE / "qa/qa.jsonl")
    raw = {str(row["qid"]): row for row in read_jsonl(SOURCE / "qa/qa.jsonl")}
    weights, selector_hash = model_weights()
    encoders = Encoders(args.device)
    report = {"status": "exploratory_post_hoc_neural_baselines_after_one_shot_test",
              "plan": "paper/EXPLORATORY_PLAN_NEURAL_BASELINES.md",
              "plan_sha256": sha256(Path("paper/EXPLORATORY_PLAN_NEURAL_BASELINES.md").read_bytes()).hexdigest(),
              "source_sha256": {name: sha256((SOURCE / rel).read_bytes()).hexdigest()
                                for name, rel in (("labels", "labels/labels.jsonl"), ("qa", "qa/qa.jsonl"))},
              "selector_model_sha256": selector_hash, "model_revisions": encoders.revisions,
              "settings": {"max_tokens": MAX_TOKENS, "rerank_depth": RERANK_DEPTH, "rrf_k": 60, "ks": KS,
                           "bootstrap_draws": DRAWS, "seed": SEED, "precision": "float16"},
              "splits": {}}
    methods = ("bm25", "tfidf", "rrf_lexical", "ontology_cover", "learned_cover", "bge_m3", "hybrid_bm25_bge",
               "rerank_bm25_top20", "rerank_hybrid_top20")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8", newline="\n") as sink:
        for split in args.splits.split(","):
            split_questions = eligible(questions, split)
            by_label = defaultdict(list)
            for q in split_questions:
                by_label[q.set_id].append(q)
            rows, timing = [], defaultdict(float)
            for set_id, label_questions in sorted(by_label.items()):
                label = labels[set_id]
                index = LabelIndex(label)
                started = perf_counter()
                passage_vectors = encoders.embed(list(label.chunks))
                timing["bge_m3_passage_encoding_s"] += perf_counter() - started
                query_vectors = encoders.embed([q.text for q in label_questions])
                for q, query in zip(label_questions, query_vectors):
                    n = len(label.chunks)
                    row = {"qid": q.qid, "set_id": set_id, "split": split, "gold": list(q.gold),
                           "source": raw[q.qid].get("source"), "question_type": raw[q.qid].get("question_type")}
                    for method in ("bm25", "tfidf", "rrf_lexical", "ontology_cover"):
                        row[method] = index.retrieve(q.text, method, max(KS))
                    # The coverage step assembles a different set for each budget; "learned_cover" is the
                    # canonical five-passage selector, and the other budgets are stored separately.
                    row["learned_cover"] = learned_retrieve(index, q.text, weights, k=5, coverage=True)
                    for k in KS:
                        row[f"learned_cover@{k}"] = learned_retrieve(index, q.text, weights, k=k, coverage=True)
                    scores = (passage_vectors @ query).tolist()
                    dense = sorted(range(n), key=lambda i: (-scores[i], i))
                    bm25_full = index.retrieve(q.text, "bm25", n)
                    hybrid = rank_rrf(bm25_full, dense)
                    started = perf_counter()
                    row["rerank_bm25_top20"] = rerank_top(encoders, q.text, label.chunks, bm25_full)[:max(KS)]
                    timing["rerank_s"] += perf_counter() - started
                    row["rerank_hybrid_top20"] = rerank_top(encoders, q.text, label.chunks, hybrid)[:max(KS)]
                    row["bge_m3"], row["hybrid_bm25_bge"] = dense[:max(KS)], hybrid[:max(KS)]
                    rows.append(row)
                    sink.write(json.dumps(row) + "\n")
            split_report = {"n_questions": len(rows), "n_labels": len(by_label),
                            "metrics": {m: metrics(rows, m) for m in methods}}
            for k in KS:  # all-gold@k for the selector uses the set it assembles at that budget.
                split_report["metrics"]["learned_cover"][f"all_gold@{k}"] = round(
                    sum(set(r["gold"]) <= set(r[f"learned_cover@{k}"]) for r in rows) / len(rows), 4)
            split_report["paired_k5"] = [paired(rows, m, base) for base in ("bm25", "learned_cover")
                                         for m in methods if m != base]
            by_source = defaultdict(list)
            for row in rows:
                by_source[row["source"]].append(row)
            split_report["all_gold@5_by_source"] = {
                source: {"n": len(group), **{m: round(sum(set(r["gold"]) <= set(r[m][:5]) for r in group) / len(group), 4)
                                            for m in methods}} for source, group in sorted(by_source.items())}
            split_report["timing_seconds"] = {key: round(value, 1) for key, value in timing.items()}
            report["splits"][split] = split_report
            print(json.dumps({"split": split, "n": len(rows),
                              "all_gold@5": {m: split_report["metrics"][m]["all_gold@5"] for m in methods}}), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output)}))


if __name__ == "__main__":
    main()
