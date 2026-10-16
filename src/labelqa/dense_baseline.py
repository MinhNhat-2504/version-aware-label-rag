"""Frozen local embedding baselines on validation labels; resumable vector cache."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

from .data import load_labels, load_questions, read_jsonl, split_label
from .evaluate import paired_cluster_bootstrap, summarize
from .retrieval import LabelIndex
from .run_local import model_digest, post_json


MODEL = "embeddinggemma:300m-qat-q4_0"
SOURCE = Path("data/upstream/FDARxBench/data")


def normalized(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def embed_missing(items: list[tuple[str, str]], cache_path: Path, digest: str,
                  batch_size: int) -> dict[str, list[float]]:
    expected = {key: text_hash(value) for key, value in items}
    vectors = {}
    if cache_path.exists():
        for row in read_jsonl(cache_path):
            key = row["key"]
            if key not in expected:
                continue
            if row["model_digest"] != digest or row["text_sha256"] != expected[key]:
                raise ValueError(f"Embedding cache identity mismatch for {key}")
            if key in vectors:
                raise ValueError(f"Duplicate embedding cache key: {key}")
            vectors[key] = row["embedding"]
    pending = [(key, value) for key, value in items if key not in vectors]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            response = post_json("/api/embed", {"model": MODEL, "input": [value for _, value in batch],
                                                 "truncate": True, "keep_alive": "30m"}, 300)
            embeddings = response.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise ValueError("Embedding service returned wrong batch size")
            for (key, value), vector in zip(batch, embeddings):
                if not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
                    raise ValueError(f"Invalid embedding for {key}")
                unit = normalized(vector)
                vectors[key] = unit
                handle.write(json.dumps({"key": key, "text_sha256": text_hash(value),
                                         "model_digest": digest, "embedding": unit}) + "\n")
            handle.flush()
            if (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(pending):
                print(json.dumps({"embedded_new": min(start + batch_size, len(pending)),
                                  "pending_initial": len(pending), "cached_total": len(vectors)}), flush=True)
    return vectors


def rank_dense(query: list[float], chunks: list[list[float]]) -> list[int]:
    if any(len(vector) != len(query) for vector in chunks):
        raise ValueError("Embedding dimensions differ")
    scores = [sum(a * b for a, b in zip(query, vector)) for vector in chunks]
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def rank_rrf(first: list[int], second: list[int]) -> list[int]:
    if len(first) != len(second) or set(first) != set(second):
        raise ValueError("RRF rankings have different passage sets")
    a = {index: rank for rank, index in enumerate(first, 1)}
    b = {index: rank for rank, index in enumerate(second, 1)}
    return sorted(first, key=lambda index: (-(1 / (60 + a[index]) + 1 / (60 + b[index])), index))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Smoke run on the first N validation multihop questions")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache", type=Path, default=Path("results/dense_validation_vectors.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/dense_validation.json"))
    parser.add_argument("--predictions", type=Path, default=Path("results/dense_validation_predictions.jsonl"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("Positive batch size and nonnegative limit required")
    labels = load_labels(SOURCE / "labels/labels.jsonl")
    questions = [q for q in load_questions(SOURCE / "qa/qa.jsonl")
                 if split_label(q.set_id) == "validation" and q.task == "multihop"
                 and 0 < len(q.gold) <= 5]
    if args.limit:
        questions = questions[:args.limit]
    label_ids = sorted({q.set_id for q in questions})
    items = [(f"passage:{set_id}:{i}", chunk) for set_id in label_ids
             for i, chunk in enumerate(labels[set_id].chunks)]
    items.extend((f"question:{q.qid}", q.text) for q in questions)
    digest = model_digest(MODEL)
    started = perf_counter()
    vectors = embed_missing(items, args.cache, digest, args.batch_size)
    embedding_seconds = round(perf_counter() - started, 3)
    indexes = {set_id: LabelIndex(labels[set_id]) for set_id in label_ids}
    chunks_by_label = {set_id: [vectors[f"passage:{set_id}:{i}"]
                                for i in range(len(labels[set_id].chunks))] for set_id in label_ids}
    outcomes = {name: [] for name in ("bm25", "dense", "bm25_dense_rrf", "dense_rerank_bm25_top20")}
    timings = {name: [] for name in outcomes}
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", encoding="utf-8") as handle:
        for q in questions:
            index = indexes[q.set_id]
            start = perf_counter()
            bm25 = index.retrieve(q.text, "bm25", len(index.label.chunks))
            timings["bm25"].append((perf_counter() - start) * 1000)
            start = perf_counter()
            dense = rank_dense(vectors[f"question:{q.qid}"], chunks_by_label[q.set_id])
            timings["dense"].append((perf_counter() - start) * 1000)
            start = perf_counter()
            rrf = rank_rrf(bm25, dense)
            timings["bm25_dense_rrf"].append((perf_counter() - start) * 1000)
            start = perf_counter()
            top20 = set(bm25[:20])
            rerank = [candidate for candidate in dense if candidate in top20][:5]
            timings["dense_rerank_bm25_top20"].append((perf_counter() - start) * 1000)
            rankings = {"bm25": bm25[:5], "dense": dense[:5], "bm25_dense_rrf": rrf[:5],
                        "dense_rerank_bm25_top20": rerank}
            for name, ranking in rankings.items():
                outcomes[name].append((q, ranking))
            handle.write(json.dumps({"qid": q.qid, "set_id": q.set_id,
                                     "gold_indices": list(q.gold), "rankings": rankings}) + "\n")
    report = {
        "status": "validation_multihop_embedding_baselines_upstream_gold_proxy",
        "model": MODEL, "model_digest": digest, "embedding_dimensions": len(next(iter(vectors.values()))),
        "embedding_api": "Ollama /api/embed", "truncate_long_inputs": True,
        "source_sha256": {"labels": sha256((SOURCE / "labels/labels.jsonl").read_bytes()).hexdigest(),
                          "qa": sha256((SOURCE / "qa/qa.jsonl").read_bytes()).hexdigest()},
        "cache_path": str(args.cache), "cache_sha256": sha256(args.cache.read_bytes()).hexdigest(),
        "predictions_path": str(args.predictions),
        "predictions_sha256": sha256(args.predictions.read_bytes()).hexdigest(),
        "n_questions": len(questions), "n_labels": len(label_ids),
        "n_passages_embedded": len(items) - len(questions), "embedding_wall_seconds": embedding_seconds,
        "metrics": {name: summarize(rows)["multihop"] for name, rows in outcomes.items()},
        "comparisons": {name + "_minus_bm25": paired_cluster_bootstrap(outcomes["bm25"], rows)
                        for name, rows in outcomes.items() if name != "bm25"},
        "mean_ranking_ms_per_question_excluding_embedding": {
            name: round(sum(values) / len(values), 3) for name, values in timings.items()},
        "limitations": [
            "The chosen model is a compact local baseline, not a pharmacology-specialized dense retriever.",
            "Long inputs may be truncated by the embedding API; embedding cost is reported separately from ranking.",
            "All-gold@5 uses upstream context indices; alternate valid evidence is not adjudicated.",
            "No test split or new-label holdout was evaluated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n": len(questions),
                      "rates": {name: row["all_gold@5"] for name, row in report["metrics"].items()},
                      "embedding_wall_seconds": embedding_seconds}))


if __name__ == "__main__":
    main()
