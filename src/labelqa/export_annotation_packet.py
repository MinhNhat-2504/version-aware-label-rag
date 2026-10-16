"""Export blinded pilot questions for independent SPL annotation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_jsonl


SOURCES = {
    "temporal": (
        Path("data/curated/temporal_pilot.jsonl"),
        "data/snapshots/recent_label_pool_10.jsonl",
    ),
    "version": (
        Path("data/curated/version_pilot_paxlovid.jsonl"),
        "data/snapshots/version_pair_paxlovid.jsonl",
    ),
}


def build(kind: str) -> list[dict]:
    path, snapshot = SOURCES[kind]
    output = []
    for row in read_jsonl(path):
        output.append({
            "qid": row["qid"], "set_id": row["set_id"],
            "spl_version": row["spl_version"],
            "source_sha256": row.get("source_sha256", row.get("xml_sha256")),
            "source_snapshot": snapshot,
            "question": row["question"],
            "reviewer_id": "", "item_decision": "",  # accept, rewrite or reject
            "independent_answer": "", "answerable_from_version": None,
            "required_claims": [], "acceptable_section_indices_per_claim": [],
            "notes": "",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(SOURCES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = build(args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "items": len(records),
                      "contains_assistant_answers": False}))


if __name__ == "__main__":
    main()
