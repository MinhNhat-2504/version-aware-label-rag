"""Add dense and BM25-dense prompts to the frozen development QA sample."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

from .data import load_labels, load_questions, split_label
from .dense_baseline import MODEL, embed_missing, rank_dense, rank_rrf
from .prompts import make_prompt
from .retrieval import LabelIndex
from .run_local import model_digest


SOURCE = Path("data/upstream/FDARxBench/data")
ROOT = Path("results")
MANIFEST = ROOT / "paired_dev_qa_manifest.json"
CACHE = ROOT / "dense_dev_qa_vectors.jsonl"
METHODS = ("dense", "bm25_dense_rrf")


def main() -> None:
    sample = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sample.get("status") != "out_of_fold_development_only_paired_prompts_no_human_claim_gold":
        raise ValueError("Frozen development sample is missing")
    labels_path = SOURCE / "labels/labels.jsonl"
    qa_path = SOURCE / "qa/qa.jsonl"
    for path in (labels_path, qa_path):
        if sha256(path.read_bytes()).hexdigest() != sample["source_sha256"][str(path)]:
            raise ValueError(f"Development source changed: {path}")
    questions = {q.qid: q for q in load_questions(qa_path) if split_label(q.set_id) == "dev"}
    labels = load_labels(labels_path)
    selected = sample["sample"]
    label_ids = sorted({row["set_id"] for row in selected})
    if len(label_ids) != len(selected):
        raise ValueError("Dense development sample has repeated labels")
    for row in selected:
        q = questions[row["qid"]]
        if q.set_id != row["set_id"] or q.task != "multihop":
            raise ValueError(f"Non-development/mismatched question: {row['qid']}")
    items = [(f"passage:{set_id}:{index}", chunk)
             for set_id in label_ids for index, chunk in enumerate(labels[set_id].chunks)]
    items.extend((f"question:{row['qid']}", questions[row["qid"]].text) for row in selected)
    digest = model_digest(MODEL)
    started = perf_counter()
    vectors = embed_missing(items, CACHE, digest, batch_size=64)
    embedding_wall_seconds = round(perf_counter() - started, 3)
    indexes = {set_id: LabelIndex(labels[set_id]) for set_id in label_ids}
    outputs = {method: ROOT / f"paired_dev_qa_{method}_prompts.jsonl" for method in METHODS}
    if any(path.exists() for path in outputs.values()):
        raise ValueError("Dense QA prompts already exist; refusing to overwrite them")
    handles = {method: path.open("w", encoding="utf-8") for method, path in outputs.items()}
    counts = {method: 0 for method in METHODS}
    try:
        for row in selected:
            q = questions[row["qid"]]
            label = labels[q.set_id]
            index = indexes[q.set_id]
            chunks = [vectors[f"passage:{q.set_id}:{i}"] for i in range(len(label.chunks))]
            dense = rank_dense(vectors[f"question:{q.qid}"], chunks)
            bm25 = index.retrieve(q.text, "bm25", len(label.chunks))
            rankings = {"dense": dense[:5], "bm25_dense_rrf": rank_rrf(bm25, dense)[:5]}
            for method, ranking in rankings.items():
                counts[method] += int(set(q.gold) <= set(ranking))
                payload = {"qid": q.qid, "set_id": q.set_id, "task": q.task,
                           "cohort": row["cohort"], "question": q.text,
                           "reference_answer": q.answer, "schema_version": 2,
                           "retrieved_indices": ranking, "gold_indices": list(q.gold),
                           **make_prompt(label, q.text, ranking)}
                handles[method].write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "status": "development_only_embedding_qa_prompts_same_frozen_sample",
        "sample_manifest_sha256": sha256(MANIFEST.read_bytes()).hexdigest(),
        "embedding_model": MODEL, "embedding_digest": digest,
        "embedding_cache_sha256": sha256(CACHE.read_bytes()).hexdigest(),
        "embedding_wall_seconds": embedding_wall_seconds,
        "n_questions": len(selected), "n_labels": len(label_ids),
        "n_passages_embedded": len(items) - len(selected),
        "all_upstream_gold_retrieved_counts": counts,
        "prompts_sha256": {method: sha256(path.read_bytes()).hexdigest()
                           for method, path in outputs.items()},
        "limitations": ["Compact local embeddings are a comparison baseline, not a domain-specific retriever.",
                        "Long source passages can be truncated by the embedding API.",
                        "This is a development sample and upstream-index retrieval proxy, not clinical support."],
    }
    output = ROOT / "paired_dev_qa_dense_manifest.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": counts,
                      "embedding_wall_seconds": embedding_wall_seconds}))


if __name__ == "__main__":
    main()
