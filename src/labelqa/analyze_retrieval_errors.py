"""Audit fixed validation retrieval predictions without changing any model or threshold."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


ROOT = Path("results")
LEARNED = ROOT / "validation_learned_predictions.jsonl"
DENSE = ROOT / "dense_validation_predictions.jsonl"
OUTPUT = ROOT / "retrieval_error_analysis.json"
METHODS = ("bm25", "ontology_cover", "learned_cover", "dense", "bm25_dense_rrf")


def main() -> None:
    learned = {row["qid"]: row for row in read_jsonl(LEARNED)}
    dense = {row["qid"]: row for row in read_jsonl(DENSE)}
    if not learned or learned.keys() != dense.keys():
        raise ValueError("Prediction files have different question sets")
    by_label = defaultdict(lambda: Counter())
    outcomes = Counter()
    overlap = Counter()
    for qid, row in learned.items():
        other = dense[qid]
        if row["set_id"] != other["set_id"] or row["gold_indices"] != other["gold_indices"]:
            raise ValueError(f"Gold or label mismatch: {qid}")
        gold = set(row["gold_indices"])
        rankings = {**row["rankings"], **other["rankings"]}
        if row["rankings"]["bm25"] != other["rankings"]["bm25"]:
            raise ValueError(f"BM25 differs between experiments: {qid}")
        complete = {name: gold <= set(rankings[name]) for name in METHODS}
        for name, success in complete.items():
            outcomes[f"{name}_complete"] += int(success)
        for comparator in ("ontology_cover", "learned_cover", "dense", "bm25_dense_rrf"):
            if complete[comparator] and not complete["bm25"]:
                outcomes[f"{comparator}_win_over_bm25"] += 1
                by_label[row["set_id"]]["wins"] += int(comparator == "learned_cover")
            if complete["bm25"] and not complete[comparator]:
                outcomes[f"{comparator}_loss_to_bm25"] += 1
                by_label[row["set_id"]]["losses"] += int(comparator == "learned_cover")
        if not any(complete.values()):
            outcomes["no_method_complete"] += 1
        if not complete["bm25"] and not complete["ontology_cover"] and complete["learned_cover"]:
            outcomes["learned_unique_beyond_bm25_and_ontology"] += 1
        if not complete["bm25"] and complete["ontology_cover"] and complete["learned_cover"]:
            outcomes["learned_and_ontology_rescue_bm25"] += 1
        overlap["bm25_learned_both_complete"] += int(complete["bm25"] and complete["learned_cover"])
        overlap["bm25_learned_both_incomplete"] += int(not complete["bm25"] and not complete["learned_cover"])
    label_deltas = sorted(({"set_id": label, "wins": counts["wins"],
                            "losses": counts["losses"], "net": counts["wins"] - counts["losses"]}
                           for label, counts in by_label.items()), key=lambda row: (-abs(row["net"]), row["set_id"]))
    report = {
        "status": "posthoc_validation_error_description_upstream_gold_proxy_only",
        "inputs_sha256": {str(path): sha256(path.read_bytes()).hexdigest() for path in (LEARNED, DENSE)},
        "questions": len(learned), "labels": len({row["set_id"] for row in learned.values()}),
        "counts": dict(sorted(outcomes.items())), "overlap": dict(overlap),
        "learned_vs_bm25_labels_with_net_gain": sum(row["net"] > 0 for row in label_deltas),
        "learned_vs_bm25_labels_with_net_loss": sum(row["net"] < 0 for row in label_deltas),
        "learned_vs_bm25_largest_absolute_label_deltas": label_deltas[:10],
        "limitation": "Exploratory validation description, not a new model-selection or confirmatory test; alternate evidence remains unadjudicated.",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": report["counts"]}))


if __name__ == "__main__":
    main()
