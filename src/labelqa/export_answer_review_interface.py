"""Build the offline answer-review interface bundle from a method-blind packet.

The bundle carries each blank packet row unchanged, the pipeline's own mechanical
classification (unusable output, refusal or answer) and the full frozen section text of
the requested SPL version, so the expert can judge omissions and refusals against the
whole label. It never reads the private method key, gold answers or retrieval reports,
and refuses a packet whose rows name a method or a reference answer.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .answer_review import FIELDS, mechanical, packet_index
from .data import portable_path, read_jsonl
from .export_reviewer_interface import ROOT, bundle_text


DEFAULT_OUTPUT = Path("answer_review_ui/dist/answer-review-data.js")
FORBIDDEN_KEYS = {"method", "reference_answer", "gold_claims", "gold_indices", "retrieved_indices",
                  "prompt_sha256", "selector_model_sha256"}


def build_payload(packet: Path, root: Path = ROOT) -> dict:
    root = root.resolve()
    packet_path = packet if packet.is_absolute() else root / packet
    rows = list(read_jsonl(packet_path))
    packet_index(rows)  # Non-empty, unique review IDs and no reviewer judgments.
    leaked = sorted({key for row in rows for key in row} & FORBIDDEN_KEYS)
    if leaked:
        raise ValueError(f"Packet exposes method or gold fields: {leaked}")
    labels: dict[str, dict] = {}
    snapshots: dict[str, dict] = {}
    items = []
    for row in rows:
        snapshot = portable_path(row["source_snapshot"])
        key = snapshot.as_posix()
        if key not in snapshots:
            path = snapshot if snapshot.is_absolute() else root / snapshot
            snapshots[key] = {"sha256": sha256(path.read_bytes()).hexdigest(),
                              "pairs": {pair["set_id"]: pair for pair in read_jsonl(path)}}
        pair = snapshots[key]["pairs"].get(row["set_id"])
        version = next((v for v in (pair or {}).get("versions", []) if v["spl_version"] == row["spl_version"]), None)
        if version is None or version["xml_sha256"] != row["xml_sha256"]:
            raise ValueError(f"Pinned source identity/hash mismatch for {row['review_id']}")
        label_key = f"{row['set_id']}_v{row['spl_version']}"
        labels.setdefault(label_key, {"title": pair["title"], "set_id": row["set_id"],
                                      "spl_version": row["spl_version"], "sections": version["sections"]})
        machine = mechanical(row)
        items.append({"row": row, "label_key": label_key,
                      "machine": {"valid_schema": machine["valid_schema"], "abstain": machine["abstain"]}})
    packet_key = portable_path(packet).as_posix()
    return {
        "schema_version": "1.0",
        "dataset": "holdout_method_blind_answer_review",
        "model_outputs_included": True,
        "method_labels_included": False,
        "packet_path": packet_key,
        "packet_sha256": sha256(packet_path.read_bytes()).hexdigest(),
        "inputs_sha256": {packet_key: sha256(packet_path.read_bytes()).hexdigest(),
                          **{name: value["sha256"] for name, value in sorted(snapshots.items())}},
        "fields": list(FIELDS),
        "items": items,
        "labels": labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True,
                        help="Blank method-blind packet from labelqa.prepare_blinded_answer_review")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    payload = build_payload(args.packet, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(bundle_text(payload).replace("window.LABELQA_REVIEW_DATA=", "window.LABELQA_ANSWER_REVIEW_DATA=", 1),
                      encoding="utf-8", newline="\n")
    kinds = {"invalid": 0, "refusal": 0, "answer": 0}
    for item in payload["items"]:
        kinds["invalid" if not item["machine"]["valid_schema"] else
              "refusal" if item["machine"]["abstain"] else "answer"] += 1
    print(json.dumps({"items": len(payload["items"]), "labels": len(payload["labels"]), "kinds": kinds,
                      "method_labels_included": False, "output": str(output)}))


if __name__ == "__main__":
    main()
