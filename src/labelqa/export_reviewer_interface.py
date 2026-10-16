"""Build the offline, model-blind reviewer interface data bundle."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from .data import portable_path, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = Path("data/review/version_holdout_reader_a_blank.jsonl")
DEFAULT_OUTPUT = Path("reviewer_ui/dist/review-data.js")
IDENTITY_FIELDS = (
    "qid", "pair_id", "set_id", "spl_version", "published_date", "xml_sha256",
    "source_url", "source_snapshot", "question", "question_origin", "annotation_status",
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def build_payload(root: Path = ROOT, packet: Path = DEFAULT_PACKET) -> dict:
    root = root.resolve()
    packet_path = packet if packet.is_absolute() else root / packet
    rows = list(read_jsonl(packet_path))
    if not rows:
        raise ValueError("Reviewer packet is empty")
    if len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate qid in reviewer packet")

    snapshot_paths = sorted({portable_path(row["source_snapshot"]) for row in rows}, key=str)
    sources: dict[tuple[str, int], dict] = {}
    titles: dict[str, str] = {}
    for relative_path in snapshot_paths:
        path = relative_path if relative_path.is_absolute() else root / relative_path
        for pair in read_jsonl(path):
            titles[pair["set_id"]] = pair["title"]
            for version in pair["versions"]:
                key = (pair["set_id"], version["spl_version"])
                if key in sources:
                    raise ValueError(f"Duplicate source version: {key}")
                sources[key] = version

    pairs: dict[str, dict] = {}
    packet_rows = []
    for original in rows:
        row = deepcopy(original)
        missing = [field for field in IDENTITY_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Missing identity fields for {row.get('qid')}: {missing}")
        source = sources.get((row["set_id"], row["spl_version"]))
        if source is None or source["xml_sha256"] != row["xml_sha256"]:
            raise ValueError(f"Pinned source identity/hash mismatch for {row['qid']}")
        if not source.get("sections"):
            raise ValueError(f"Pinned source has no sections for {row['qid']}")
        item = {**row, "title": titles[row["set_id"]], "sections": source["sections"]}
        pair = pairs.setdefault(row["pair_id"], {
            "pair_id": row["pair_id"], "set_id": row["set_id"],
            "title": titles[row["set_id"]], "question": row["question"], "items": [],
        })
        if pair["question"] != row["question"] or pair["set_id"] != row["set_id"]:
            raise ValueError(f"Inconsistent pair metadata for {row['pair_id']}")
        pair["items"].append(item)
        packet_rows.append({key: deepcopy(row[key]) for key in row})

    ordered_pairs = list(pairs.values())
    for pair in ordered_pairs:
        pair["items"].sort(key=lambda item: item["spl_version"])
        if len(pair["items"]) != 2:
            raise ValueError(f"Expected exactly two versions for {pair['pair_id']}")

    inputs = {str(packet): file_sha256(packet_path)}
    for relative_path in snapshot_paths:
        path = relative_path if relative_path.is_absolute() else root / relative_path
        inputs[str(relative_path)] = file_sha256(path)
    return {
        "schema_version": "1.0",
        "dataset": "sealed_version_holdout_human_review",
        "model_outputs_included": False,
        "review_status": "candidate_pending_two_independent_readers",
        "identity_fields": list(IDENTITY_FIELDS),
        "annotation_fields": [
            "reviewer_id", "item_decision", "independent_answer",
            "answerable_from_version", "required_claims",
            "acceptable_section_indices_per_claim", "version_change_judgment",
            "clinical_significance", "notes", "acceptable_evidence_spans_per_claim",
        ],
        "inputs_sha256": inputs,
        "packet_rows": packet_rows,
        "pairs": ordered_pairs,
    }


def bundle_text(payload: dict) -> str:
    """Return a deterministic browser script for drift checks and export."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return f"window.LABELQA_REVIEW_DATA={encoded};\n"


def write_bundle(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(bundle_text(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    payload = build_payload(root, args.packet)
    write_bundle(payload, output)
    print(json.dumps({
        "pairs": len(payload["pairs"]),
        "items": len(payload["packet_rows"]),
        "model_outputs_included": payload["model_outputs_included"],
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
