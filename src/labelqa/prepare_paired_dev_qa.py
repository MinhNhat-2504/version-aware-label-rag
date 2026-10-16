"""Freeze paired out-of-fold development QA prompts before local generation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import load_labels, load_questions, read_jsonl, split_label
from .prompts import make_prompt


SOURCE = Path("data/upstream/FDARxBench/data")
CROSSFIT = Path("results/dev_learned_crossfit_predictions.jsonl")
ROOT = Path("results")
METHODS = {"bm25": "bm25", "learned_cover": "learned_cover"}


def select_rows(rows: list[dict], representative_n: int = 24,
                challenge_each: int = 8) -> list[tuple[str, dict]]:
    """One question per label; challenge strata use retrieval gold, never model answers."""
    ordered = sorted(rows, key=lambda row: (sha256(row["qid"].encode()).hexdigest(), row["qid"]))
    used_labels: set[str] = set()
    selected: list[tuple[str, dict]] = []

    def add(cohort: str, candidates: list[dict], target: int) -> None:
        count = 0
        for row in candidates:
            if row["set_id"] in used_labels:
                continue
            selected.append((cohort, row))
            used_labels.add(row["set_id"])
            count += 1
            if count == target:
                return
        raise ValueError(f"Only {count}/{target} label-distinct questions for {cohort}")

    add("representative_hash_sample", ordered, representative_n)
    wins, losses = [], []
    for row in ordered:
        gold = set(row["gold_indices"])
        baseline = gold <= set(row["rankings"]["bm25"])
        candidate = gold <= set(row["rankings"]["learned_cover"])
        if candidate and not baseline:
            wins.append(row)
        elif baseline and not candidate:
            losses.append(row)
    add("retrieval_challenge_learned_win", wins, challenge_each)
    add("retrieval_challenge_bm25_win", losses, challenge_each)
    return selected


def main() -> None:
    labels_path = SOURCE / "labels/labels.jsonl"
    qa_path = SOURCE / "qa/qa.jsonl"
    rows = list(read_jsonl(CROSSFIT))
    if len(rows) != len({row["qid"] for row in rows}):
        raise ValueError("Duplicate crossfit question ID")
    questions = {q.qid: q for q in load_questions(qa_path) if split_label(q.set_id) == "dev"}
    labels = load_labels(labels_path)
    selected = select_rows(rows)
    outputs = {name: ROOT / f"paired_dev_qa_{name}_prompts.jsonl" for name in METHODS}
    handles = {name: path.open("w", encoding="utf-8") for name, path in outputs.items()}
    sample = []
    try:
        for cohort, row in selected:
            qid = row["qid"]
            q = questions[qid]
            if q.task != "multihop" or q.set_id != row["set_id"] or list(q.gold) != row["gold_indices"]:
                raise ValueError(f"Crossfit question identity mismatch: {qid}")
            if len(q.gold) > 5:
                raise ValueError(f"Ineligible gold count: {qid}")
            sample.append({"qid": qid, "set_id": q.set_id, "cohort": cohort,
                           "bm25_complete_gold": set(q.gold) <= set(row["rankings"]["bm25"]),
                           "learned_complete_gold": set(q.gold) <= set(row["rankings"]["learned_cover"])})
            for name, key in METHODS.items():
                ranking = row["rankings"][key]
                if len(ranking) != 5 or len(set(ranking)) != 5:
                    raise ValueError(f"Expected five unique passages: {qid} {name}")
                payload = {"qid": qid, "set_id": q.set_id, "task": q.task,
                           "cohort": cohort, "question": q.text,
                           "reference_answer": q.answer, "schema_version": 2,
                           "retrieved_indices": ranking, "gold_indices": list(q.gold),
                           **make_prompt(labels[q.set_id], q.text, ranking)}
                handles[name].write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    manifest = {
        "status": "out_of_fold_development_only_paired_prompts_no_human_claim_gold",
        "selection": "24 hash-ordered label-distinct questions, then 8 each with learned/BM25 retrieval proxy advantage; all disjoint labels",
        "representative_n": 24, "challenge_each_n": 8, "n_questions": len(sample),
        "source_sha256": {str(path): sha256(path.read_bytes()).hexdigest()
                          for path in (labels_path, qa_path, CROSSFIT)},
        "prompts_sha256": {name: sha256(path.read_bytes()).hexdigest() for name, path in outputs.items()},
        "sample": sample,
        "limitations": ["Challenge strata are selected using upstream retrieval gold and are not prevalence estimates.",
                        "Development crossfit rankings exclude each held label from training, but this is not an external test.",
                        "Gold indices and lexical answer similarity do not certify clinical answer or citation support."],
    }
    path = ROOT / "paired_dev_qa_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(path), "questions": len(sample),
                      "prompts": {name: str(path) for name, path in outputs.items()}}))


if __name__ == "__main__":
    main()
