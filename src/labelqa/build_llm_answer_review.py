"""Convert a disclosed LLM answer reviewer's judgments into a checked answer-review file.

The specification holds, per opaque ``review_id`` of the method-blind packet, the same
fields a human answer reviewer fills. The LLM must work from the blank packet only: never
the private method key, gold answers or retrieval summaries. The output uses an ``llm_``
reviewer ID, passes the same validation as a human file and gets a provenance receipt.
It is one annotation, never a resolution; the human expert signs every final item.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .answer_review import FIELDS, LLM_ANSWER_STATUS, packet_index, validate_review
from .data import read_jsonl
from .review_resolution import is_llm


def build_rows(spec: dict, packet: list[dict]) -> list[dict]:
    annotator = spec.get("annotator_id")
    if not is_llm(annotator):
        raise ValueError("LLM answer reviews must use an llm_ reviewer ID")
    if not spec.get("disclosures"):
        raise ValueError("An LLM answer review needs its disclosures")
    blank = packet_index(packet)
    items = spec.get("items")
    if not isinstance(items, dict) or set(items) != set(blank):
        raise ValueError("LLM answer review must cover exactly the packet review IDs")
    rows = []
    for row in packet:
        judged = items[row["review_id"]]
        unknown = set(judged) - set(FIELDS)
        if unknown:
            raise ValueError(f"Unknown answer-review fields for {row['review_id']}: {sorted(unknown)}")
        review = {**row, "reviewer_id": annotator, **{field: judged.get(field, row[field]) for field in FIELDS}}
        validate_review(row, review)
        rows.append(review)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.output.with_suffix(".provenance.json")
    if args.output.exists() or receipt_path.exists():
        parser.error("Answer-review output exists; write a new versioned file")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    rows = build_rows(spec, list(read_jsonl(args.packet)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8", newline="\n")
    receipt = {
        "status": LLM_ANSWER_STATUS, "annotator_id": spec["annotator_id"], "model": spec.get("model"),
        "created_utc": datetime.now(timezone.utc).isoformat(), "disclosures": spec["disclosures"],
        "spec_sha256": sha256(args.spec.read_bytes()).hexdigest(),
        "packet_sha256": sha256(args.packet.read_bytes()).hexdigest(),
        "annotation_sha256": sha256(args.output.read_bytes()).hexdigest(),
        "items": len(rows),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                            newline="\n")
    print(json.dumps({"output": str(args.output), "items": len(rows), "provenance": str(receipt_path)}))


if __name__ == "__main__":
    main()
