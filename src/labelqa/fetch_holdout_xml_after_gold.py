"""Fetch pinned holdout XML only after independently adjudicated gold is frozen."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import portable_path, read_jsonl
from .prepare_holdout_eval import DEFAULT_GOLD, load_manual_gold
from .recent import parse_sections
from .version_pairs import download_version_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=Path("results/pinned_xml_holdout_after_gold"))
    args = parser.parse_args()
    rows, _ = load_manual_gold(args.gold)  # This must precede source/network access.
    versions = {}
    for row in rows:
        source = portable_path(row["source_snapshot"])
        if source not in versions:
            versions[source] = {pair["set_id"]: pair for pair in read_jsonl(source)}
    pending = {}
    for row in rows:
        key = (row["set_id"], row["spl_version"])
        pair = versions[portable_path(row["source_snapshot"])][row["set_id"]]
        pinned = next((version for version in pair["versions"]
                       if version["spl_version"] == row["spl_version"]), None)
        if pinned is None or pinned["xml_sha256"] != row["xml_sha256"]:
            raise ValueError(f"Gold/source identity mismatch for {row['qid']}")
        pending[key] = pinned
    verified = []
    for (set_id, version), pinned in sorted(pending.items()):
        path = args.output_dir / f"{set_id}_v{version}.xml"
        if path.exists():
            xml_bytes = path.read_bytes()
            zip_digest = pinned["zip_sha256"]
        else:
            xml_bytes, zip_digest, _ = download_version_xml(set_id, version)
        if zip_digest != pinned["zip_sha256"] or sha256(xml_bytes).hexdigest() != pinned["xml_sha256"]:
            raise ValueError(f"Pinned ZIP/XML checksum mismatch for {set_id} v{version}")
        if parse_sections(xml_bytes) != pinned["sections"]:
            raise ValueError(f"Pinned section content mismatch for {set_id} v{version}")
        if not path.exists():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(xml_bytes)
        verified.append({"set_id": set_id, "spl_version": version,
                         "xml_sha256": pinned["xml_sha256"], "path": str(path)})
    print(json.dumps({"verified_versions": len(verified), "items": verified}))


if __name__ == "__main__":
    main()
