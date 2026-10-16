"""Build an unannotated post-benchmark human-prescription-label pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from .data import load_questions


BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
NS = {"hl7": "urn:hl7-org:v3"}


def get_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "LabelEvidenceResearch/0.1 (academic research)"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def parse_sections(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    sections = []
    for number, section in enumerate(root.findall(".//hl7:section", NS)):
        title_node = section.find("hl7:title", NS)
        text_node = section.find("hl7:text", NS)
        code_node = section.find("hl7:code", NS)
        if text_node is None:
            continue
        text = " ".join(" ".join(text_node.itertext()).split())
        if not text:
            continue
        sections.append({
            "section_index": number,
            "title": " ".join(" ".join(title_node.itertext()).split()) if title_node is not None else "",
            "code": code_node.get("code") if code_node is not None else None,
            "text": text,
        })
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path("data/upstream/FDARxBench/data/qa/qa.jsonl"))
    parser.add_argument("--since", default="2026-09-03", help="DailyMed publication date, YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=Path("results/recent_label_pool.jsonl"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    benchmark_ids = {question.set_id for question in load_questions(args.qa)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fetched, failures, page = 0, 0, 1
    with args.output.open("w", encoding="utf-8") as handle:
        while fetched < args.limit:
            params = urlencode({
                "doctype": "34391-3",  # FDA code for human prescription labels.
                "published_date": args.since,
                "published_date_comparison": "gte",
                "pagesize": 100,
                "page": page,
            })
            listing_url = f"{BASE}.json?{params}"
            listing = json.loads(get_bytes(listing_url))
            entries = listing.get("data", [])
            if not entries:
                break
            for entry in entries:
                set_id = entry.get("setid")
                if not set_id or set_id in benchmark_ids:
                    continue
                source_url = f"{BASE}/{set_id}.xml"
                try:
                    xml_bytes = get_bytes(source_url)
                    sections = parse_sections(xml_bytes)
                    if not sections:
                        raise ValueError("No nonempty SPL sections")
                except (OSError, ValueError, ET.ParseError) as exc:
                    failures += 1
                    print(f"skip {set_id}: {exc}")
                    continue
                record = {
                    "set_id": set_id,
                    "title": entry.get("title"),
                    "spl_version": entry.get("spl_version"),
                    "published_date": entry.get("published_date"),
                    "doctype": "34391-3",
                    "source_url": source_url,
                    "source_sha256": sha256(xml_bytes).hexdigest(),
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "sections": sections,
                    "annotation_status": "unannotated",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                fetched += 1
                if fetched >= args.limit:
                    break
                if args.pause:
                    time.sleep(args.pause)
            page += 1
            if page > int(listing.get("metadata", {}).get("total_pages", page)):
                break
    print(json.dumps({"output": str(args.output), "fetched": fetched, "failures": failures}))


if __name__ == "__main__":
    main()
