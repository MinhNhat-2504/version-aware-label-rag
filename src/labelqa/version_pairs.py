"""Freeze consecutive DailyMed SPL versions for exploratory temporal QA."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from .data import read_jsonl
from .recent import NS, parse_sections


SELECTED = (
    "00623958-a8fe-47d1-a0d2-0aa4e2f7966e",  # GASTROCROM
    "01683314-5996-dbe4-e063-6394a90aa07a",  # atorvastatin
    "017bc6a3-5b3d-2381-e063-6394a90abece",  # olmesartan/HCTZ
)


def download_version_xml(set_id: str, version: int) -> tuple[bytes, str, str]:
    url = "https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?" + urlencode({
        "type": "zip", "setid": set_id, "version": version,
    })
    request = Request(url, headers={"User-Agent": "LabelEvidenceResearch/0.1 (academic research)"})
    with urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    with ZipFile(BytesIO(archive_bytes)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        if len(members) != 1:
            raise ValueError(f"Expected one XML for {set_id} v{version}: {members}")
        xml_bytes = archive.read(members[0])
    root = ET.fromstring(xml_bytes)
    set_node = root.find("hl7:setId", NS)
    version_node = root.find("hl7:versionNumber", NS)
    embedded_set = set_node.get("root") if set_node is not None else None
    embedded_version = version_node.get("value") if version_node is not None else None
    if embedded_set != set_id or embedded_version != str(version):
        raise ValueError(f"Archive identity mismatch: {set_id} v{version}; XML {embedded_set} v{embedded_version}")
    return xml_bytes, sha256(archive_bytes).hexdigest(), url


def get_version(set_id: str, version: int) -> dict:
    xml_bytes, zip_digest, url = download_version_xml(set_id, version)
    return {
        "spl_version": version, "source_url": url,
        "xml_sha256": sha256(xml_bytes).hexdigest(),
        "zip_sha256": zip_digest,
        "sections": parse_sections(xml_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=Path("data/snapshots/dailymed_history_10.jsonl"))
    parser.add_argument("--set-id", action="append", dest="set_ids",
                        help="Fetch live history for this SET ID; repeat to select several labels")
    parser.add_argument("--output", type=Path, default=Path("data/snapshots/version_pairs_3.jsonl"))
    args = parser.parse_args()
    if args.set_ids:
        history = {}
        for set_id in args.set_ids:
            url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}/history.json"
            with urlopen(Request(url, headers={"User-Agent": "LabelEvidenceResearch/0.1 (academic research)"}), timeout=30) as response:
                payload = json.load(response)["data"]
            history[set_id] = {"title": payload["spl"]["title"], "versions": payload["history"]}
    else:
        history = {row["set_id"]: row for row in read_jsonl(args.history)}
    records = []
    for set_id in args.set_ids or SELECTED:
        item = history[set_id]
        versions = sorted(item["versions"], key=lambda row: row["spl_version"], reverse=True)[:2]
        if len(versions) != 2:
            raise ValueError(f"No consecutive version pair for {set_id}")
        records.append({
            "set_id": set_id, "title": item["title"],
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "versions": [
                {"published_date": version["published_date"],
                 **get_version(set_id, version["spl_version"])}
                for version in sorted(versions, key=lambda row: row["spl_version"])
            ],
            "annotation_status": "unannotated_version_pair",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "pairs": len(records),
                      "section_counts": [[len(v["sections"]) for v in row["versions"]] for row in records]}))


if __name__ == "__main__":
    main()
