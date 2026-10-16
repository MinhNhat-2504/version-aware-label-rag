"""Fetch pinned holdout XML for human source display and evidence locators only.

Readers must inspect table layout and record exact prose or table-row locators before
gold is locked. This tool downloads each candidate version, verifies the ZIP and XML
SHA-256 and the parsed section text against the frozen snapshot, and stores the XML in
a review-only directory. It runs no retrieval, prompt construction or model call.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

from .data import portable_path, read_jsonl
from .recent import parse_sections
from .version_pairs import download_version_xml


DEFAULT_CANDIDATES = Path("data/holdout/version_question_candidates_expanded.jsonl")
DEFAULT_OUTPUT = Path("results/pinned_xml_holdout_review")
PAUSE_SECONDS = 0.3


def pinned_versions(candidates: Path) -> dict[tuple[str, int], dict]:
    """Return the frozen snapshot record for every candidate SPL version."""
    snapshots: dict[Path, dict] = {}
    pinned = {}
    for row in read_jsonl(candidates):
        source = portable_path(row["source_snapshot"])
        if source not in snapshots:
            snapshots[source] = {pair["set_id"]: pair for pair in read_jsonl(source)}
        version = next((item for item in snapshots[source][row["set_id"]]["versions"]
                        if item["spl_version"] == row["spl_version"]), None)
        if version is None or version["xml_sha256"] != row["xml_sha256"]:
            raise ValueError(f"Candidate/source identity mismatch: {row['qid']}")
        pinned[(row["set_id"], row["spl_version"])] = version
    return pinned


def verify_xml(xml_bytes: bytes, pinned: dict, label: str) -> None:
    if sha256(xml_bytes).hexdigest() != pinned["xml_sha256"]:
        raise ValueError(f"Pinned XML checksum mismatch: {label}")
    if parse_sections(xml_bytes) != pinned["sections"]:
        raise ValueError(f"Pinned section text differs from frozen snapshot: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verified, downloaded = [], 0
    for (set_id, number), pinned in sorted(pinned_versions(args.candidates).items()):
        label = f"{set_id}_v{number}"
        path = args.output_dir / f"{label}.xml"
        if path.exists():
            xml_bytes = path.read_bytes()
        else:
            xml_bytes, zip_digest, _ = download_version_xml(set_id, number)
            if zip_digest != pinned["zip_sha256"]:
                raise ValueError(f"Pinned ZIP checksum mismatch: {label}")
            downloaded += 1
            time.sleep(PAUSE_SECONDS)
        verify_xml(xml_bytes, pinned, label)
        if not path.exists():
            path.write_bytes(xml_bytes)
        verified.append({"set_id": set_id, "spl_version": number, "xml_sha256": pinned["xml_sha256"]})
    receipt = {"status": "review_display_xml_verified_no_retrieval_or_model",
               "verified_utc": datetime.now(timezone.utc).isoformat(),
               "candidates_sha256": sha256(args.candidates.read_bytes()).hexdigest(),
               "versions": verified}
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified_versions": len(verified), "downloaded": downloaded,
                      "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
