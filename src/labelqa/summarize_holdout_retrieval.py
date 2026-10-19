"""Prespecified paired retrieval summary; requires compiled manual holdout gold."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from random import Random

from .data import portable_path
from .holdout_reporting import cohort_for, sample_flow
from .prepare_holdout_eval import DEFAULT_GOLD, load_manual_gold, model_weights
from .review_resolution import review_design


METHODS = ("bm25", "learned_cover", "ontology_cover")
BOOTSTRAP_SEED = 20261016
BOOTSTRAP_REPS = 10_000


def read_prompts(path: Path, method: str, gold_hash: str,
                 gold_manifest_hash: str, expected_model_hash: str | None,
                 execution_lock_hash: str) -> dict[str, dict]:
    receipt_path = path.with_suffix(".manifest.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "holdout_prompts_after_manual_gold"
            or receipt.get("method") != method or receipt.get("gold_sha256") != gold_hash
            or receipt.get("gold_manifest_sha256") != gold_manifest_hash
            or receipt.get("execution_lock_sha256") != execution_lock_hash
            or receipt.get("model_sha256") != expected_model_hash
            or receipt.get("prompts_sha256") != sha256(path.read_bytes()).hexdigest()):
        raise ValueError(f"Unverified holdout prompt provenance: {method}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != receipt.get("n") or len({x["qid"] for x in rows}) != len(rows):
        raise ValueError(f"Duplicate or missing prompt identities: {method}")
    if any(x.get("method") != method or x.get("schema_version") != 2
           or x.get("selector_model_sha256") != receipt.get("model_sha256") for x in rows):
        raise ValueError(f"Unexpected method or schema: {method}")
    return {x["qid"]: x for x in rows}


def paired_summary(items: list[dict], left: str, right: str,
                   seed: int = BOOTSTRAP_SEED, repetitions: int = BOOTSTRAP_REPS) -> dict:
    """Bootstrap label pairs, retaining all accepted version items per pair."""
    if not items or repetitions < 1:
        raise ValueError("Nonempty paired items and positive repetitions required")
    clusters: dict[str, list[dict]] = {}
    cells = {"both": 0, "left_only": 0, "right_only": 0, "neither": 0}
    for row in items:
        if not isinstance(row.get(left), bool) or not isinstance(row.get(right), bool):
            raise ValueError("Paired outcomes must be Boolean")
        clusters.setdefault(row["set_id"], []).append(row)
        a, b = row[left], row[right]
        cells["both" if a and b else "left_only" if a else "right_only" if b else "neither"] += 1
    n = len(items)
    delta = (cells["left_only"] - cells["right_only"]) / n
    pair_ids = sorted(clusters)
    rng = Random(seed)
    samples = []
    for _ in range(repetitions):
        sampled = [row for _ in pair_ids for row in clusters[rng.choice(pair_ids)]]
        samples.append(sum(row[left] - row[right] for row in sampled) / len(sampled))
    samples.sort()
    def percentile(p: float) -> float:
        index = (len(samples) - 1) * p
        lo = int(index)
        return samples[lo] + (samples[min(lo + 1, len(samples) - 1)] - samples[lo]) * (index - lo)
    return {"left": left, "right": right, "items": n, "label_pairs": len(pair_ids),
            "paired_cells": cells, "left_success": cells["both"] + cells["left_only"],
            "right_success": cells["both"] + cells["right_only"],
            "difference_percentage_points": round(100 * delta, 4),
            "pair_bootstrap_95ci_percentage_points": [round(100 * percentile(.025), 4),
                                                       round(100 * percentile(.975), 4)],
            "bootstrap_seed": seed, "bootstrap_repetitions": repetitions}


def summarize(gold: list[dict], prompts: dict[str, dict[str, dict]]) -> dict:
    accepted = {row["qid"]: row for row in gold}
    if not accepted or any(set(rows) != set(accepted) for rows in prompts.values()):
        raise ValueError("Prompt and manually accepted QID sets differ")
    items = []
    for qid, row in accepted.items():
        cohort = cohort_for(row["source_snapshot"])
        outcomes = {"qid": qid, "set_id": row["set_id"], "spl_version": row["spl_version"],
                    "answerable": row["answerable"], "cohort": cohort}
        for method in METHODS:
            prompt = prompts[method][qid]
            for key in ("qid", "set_id", "spl_version", "question", "xml_sha256"):
                if prompt.get(key) != row.get(key):
                    raise ValueError(f"Gold/prompt identity mismatch: {qid}, {method}, {key}")
            # Frozen candidates written on Windows keep backslashes; compare normalized paths.
            if portable_path(prompt.get("source_snapshot", "")).as_posix() != portable_path(row["source_snapshot"]).as_posix():
                raise ValueError(f"Gold/prompt identity mismatch: {qid}, {method}, source_snapshot")
            if (prompt.get("task") == "refusal") != (not row["answerable"]):
                raise ValueError(f"Gold/prompt answerability mismatch: {qid}, {method}")
            retrieved = prompt.get("retrieved_indices")
            if (not isinstance(retrieved, list) or len(retrieved) > 5
                    or any(type(index) is not int or index < 0 for index in retrieved)
                    or len(set(retrieved)) != len(retrieved)):
                raise ValueError(f"Invalid passage budget or duplicate rank: {qid}, {method}")
            if not row["answerable"] and prompt.get("gold_claims"):
                raise ValueError(f"Unanswerable item has positive gold passage groups: {qid}, {method}")
        if not row["answerable"]:
            continue
        gold_groups = None
        for method in METHODS:
            prompt = prompts[method][qid]
            if not prompt.get("gold_claims") or len(prompt["gold_claims"]) != len(row["claims"]):
                raise ValueError(f"Missing answerable gold passage groups: {qid}, {method}")
            groups = [set(group) for group in prompt["gold_claims"]]
            if any(not group for group in groups) or (gold_groups is not None and groups != gold_groups):
                raise ValueError(f"Different accepted evidence groups across methods: {qid}")
            gold_groups = groups
            outcomes[method] = all(set(prompt["retrieved_indices"]) & group for group in groups)
        items.append(outcomes)
    if not items:
        raise ValueError("No accepted answerable version items")
    by_cohort = {}
    for cohort in ("consecutive_pairs", "text_change_enriched"):
        subset = [item for item in items if item["cohort"] == cohort]
        by_cohort[cohort] = {"answerable_items": len(subset),
                             "label_pairs": len({item["set_id"] for item in subset}),
                             "success_counts": {method: sum(item[method] for item in subset)
                                                for method in METHODS},
                             "learned_vs_bm25": paired_summary(subset, "learned_cover", "bm25")
                             if subset else None,
                             "learned_vs_ontology": paired_summary(subset, "learned_cover", "ontology_cover")
                             if subset else None}
    return {"status": "after_manual_gold_retrieval_only_no_answer_support_claim",
            "answerable_items": len(items), "unanswerable_items": len(gold) - len(items),
            "learned_vs_bm25": paired_summary(items, "learned_cover", "bm25"),
            "learned_vs_ontology": paired_summary(items, "learned_cover", "ontology_cover"),
            "by_cohort": by_cohort, "per_item": items}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--execution-lock", type=Path, default=Path("paper/HOLDOUT_EXECUTION_LOCK.json"))
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a holdout report")
    gold, receipt = load_manual_gold(args.gold)  # Gate before any prompt is read.
    from .lock_holdout_execution import verify_lock
    lock, lock_hash = verify_lock(args.execution_lock, gold, args.gold, receipt)
    paths = {method: getattr(args, method) for method in METHODS}
    if any(paths[method] != portable_path(lock["prompt_paths"][method]) for method in METHODS):
        raise ValueError("Retrieval summary prompt paths differ from execution lock")
    gold_manifest_hash = sha256(args.gold.with_suffix(".manifest.json").read_bytes()).hexdigest()
    selector_hash = model_weights()[1]
    prompts = {method: read_prompts(path, method, receipt["gold_sha256"], gold_manifest_hash,
                                    selector_hash if method == "learned_cover" else None,
                                    lock_hash)
               for method, path in paths.items()}
    report = summarize(gold, prompts)
    report["sample_flow"] = sample_flow(gold, receipt)
    report["gold_review_design"] = review_design(receipt)
    report["gold_sha256"] = receipt["gold_sha256"]
    report["execution_lock_sha256"] = lock_hash
    report["prompt_sha256"] = {method: sha256(path.read_bytes()).hexdigest()
                               for method, path in paths.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "answerable_items": report["answerable_items"]}))


if __name__ == "__main__":
    main()
