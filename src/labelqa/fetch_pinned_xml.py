"""Download development SPL XML only when it matches an existing pinned hash."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .version_pairs import download_version_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/snapshots/version_pair_paxlovid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pinned_xml_development"))
    args = parser.parse_args()
    source_rows = list(read_jsonl(args.source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for pair in source_rows:
        for version in pair["versions"]:
            path = args.output_dir / f"{pair['set_id']}_v{version['spl_version']}.xml"
            if path.exists():
                xml_bytes = path.read_bytes()
            else:
                xml_bytes, zip_digest, _ = download_version_xml(pair["set_id"], version["spl_version"])
                if zip_digest != version["zip_sha256"]:
                    raise ValueError(f"ZIP checksum mismatch for {path}")
            digest = sha256(xml_bytes).hexdigest()
            if digest != version["xml_sha256"]:
                raise ValueError(f"XML checksum mismatch for {path}")
            if not path.exists():
                path.write_bytes(xml_bytes)
            downloaded.append({"set_id": pair["set_id"], "spl_version": version["spl_version"],
                               "xml_sha256": digest, "path": str(path)})
    print(json.dumps({"verified_xml": downloaded}))


if __name__ == "__main__":
    main()
