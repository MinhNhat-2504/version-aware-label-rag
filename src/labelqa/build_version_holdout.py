"""Freeze a version-pair holdout before any retrieval or generation is run on it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlencode

from .data import load_questions, read_jsonl
from .recent import BASE, get_bytes
from .version_pairs import get_version


LISTING_PARAMS = {
    "doctype": "34391-3", "published_date": "2026-09-03",
    "published_date_comparison": "gte", "pagesize": 100, "page": 1,
}
LISTING_URL = f"{BASE}.json?{urlencode(LISTING_PARAMS)}"
PRIOR_SNAPSHOTS = (
    Path("data/snapshots/recent_label_pool_10.jsonl"),
    Path("data/snapshots/version_pair_paxlovid.jsonl"),
    Path("data/snapshots/version_pairs_3.jsonl"),
)


def prior_ids() -> set[str]:
    ids = {q.set_id for q in load_questions(Path("data/upstream/FDARxBench/data/qa/qa.jsonl"))}
    for path in PRIOR_SNAPSHOTS:
        ids.update(row["set_id"] for row in read_jsonl(path))
    return ids


def freeze(limit: int) -> tuple[list[dict], dict]:
    listing_bytes = get_bytes(LISTING_URL)
    listing = json.loads(listing_bytes)
    excluded = prior_ids()
    candidates = [e for e in listing["data"] if e.get("setid") not in excluded
                  and int(e.get("spl_version") or 0) >= 2]
    selected, screening = [], []
    for entry in candidates:
        if len(selected) == limit:
            break
        set_id = entry["setid"]
        history_url = f"{BASE}/{set_id}/history.json"
        try:
            history = json.loads(get_bytes(history_url))["data"]["history"]
            ordered = sorted(history, key=lambda v: int(v["spl_version"]), reverse=True)
            if len(ordered) < 2:
                raise ValueError("fewer than two history versions")
            newest, prior = ordered[:2]
            if int(newest["spl_version"]) != int(prior["spl_version"]) + 1:
                raise ValueError("latest two versions are not consecutive")
            versions = []
            for meta in (prior, newest):
                versions.append({"published_date": meta["published_date"],
                                 **get_version(set_id, int(meta["spl_version"]))})
            selected.append({
                "set_id": set_id, "title": entry["title"],
                "listed_published_date": entry["published_date"],
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "versions": versions, "annotation_status": "sealed_pending_human_review",
            })
            screening.append({"set_id": set_id, "decision": "included",
                              "versions": [v["spl_version"] for v in versions]})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            screening.append({"set_id": set_id, "decision": "excluded_download_or_history",
                              "reason": str(exc)[:200]})
    manifest = {
        "status": "sealed_sources_no_model_exposure_pending_human_questions_and_adjudication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "listing_url": LISTING_URL,
        "listing_sha256": sha256(listing_bytes).hexdigest(),
        "listing_metadata": listing.get("metadata", {}),
        "sampling_rule": "In API page-1 order, exclude all FDARxBench QA set IDs and all previously snapshotted set IDs; require SPL version >=2 and two most recent history versions to be consecutive; include first N successfully downloaded pairs regardless of content change. Do not replace a pair due to its clinical content.",
        "target_pairs": limit, "selected_pairs": len(selected),
        "screening": screening,
        "prohibitions": ["No model outputs, prompt tuning, or retrieval development on these sources before evaluation lock.",
                         "Assistant-drafted questions or judgments are not independent human annotations."],
    }
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--sources", type=Path, default=Path("data/snapshots/version_holdout_sources.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/holdout/version_test_manifest.json"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.sources.exists() or args.manifest.exists():
        parser.error("Holdout files already exist; refusing to overwrite a frozen sample")
    records, manifest = freeze(args.limit)
    if len(records) != args.limit:
        raise RuntimeError(f"Only {len(records)}/{args.limit} pairs available; holdout was not written")
    args.sources.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.sources.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    manifest["sources_path"] = str(args.sources)
    manifest["sources_sha256"] = sha256(args.sources.read_bytes()).hexdigest()
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(records), "screened": len(manifest["screening"]),
                      "sources": str(args.sources), "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
