"""Collect official DailyMed SPL version histories for a temporal study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data import load_questions


API = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}/history.json"


def fetch_history(set_id: str, timeout: float = 20.0) -> dict:
    url = API.format(set_id=set_id)
    request = Request(url, headers={"User-Agent": "LabelEvidenceResearch/0.1 (academic research)"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    data = payload.get("data", {})
    return {
        "set_id": set_id,
        "title": data.get("spl", {}).get("title"),
        "versions": data.get("history", []),
        "source_url": url,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, default=Path("data/upstream/FDARxBench/data/qa/qa.jsonl"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.2, help="Seconds between requests")
    parser.add_argument("--output", type=Path, default=Path("results/dailymed_history.jsonl"))
    args = parser.parse_args()
    ids = sorted({question.set_id for question in load_questions(args.qa) if question.set_id})
    if args.limit:
        ids = ids[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "error": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        for set_id in ids:
            try:
                result = fetch_history(set_id)
                counts["ok"] += 1
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                result = {
                    "set_id": set_id,
                    "source_url": API.format(set_id=set_id),
                    "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                }
                counts["error"] += 1
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if args.pause:
                time.sleep(args.pause)
    print(json.dumps({"output": str(args.output), "n": len(ids), **counts}))


if __name__ == "__main__":
    main()
