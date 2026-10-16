"""Freeze a predeclared clinically changed version-pair cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re

from .build_version_holdout import LISTING_URL, prior_ids
from .data import read_jsonl
from .recent import BASE, get_bytes
from .version_pairs import get_version


CLINICAL = re.compile(r"\b(INDICATIONS|DOSAGE|CONTRAINDICATIONS|WARNINGS|PRECAUTIONS|ADVERSE REACTIONS|DRUG INTERACTIONS|USE IN SPECIFIC POPULATIONS|PREGNANCY|LACTATION|PEDIATRIC USE|GERIATRIC USE|OVERDOSAGE)\b", re.I)


def changed_clinical_sections(old: dict, new: dict) -> list[dict]:
    old_texts = {(s["code"], s["title"]): s["text"] for s in old["sections"]}
    changes = []
    for section in new["sections"]:
        title = section["title"]
        if CLINICAL.search(title) and old_texts.get((section["code"], title)) != section["text"]:
            changes.append({"section_index": section["section_index"], "title": title,
                            "code": section["code"], "old_present": (section["code"], title) in old_texts})
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=4)
    parser.add_argument("--max-screen", type=int, default=50)
    parser.add_argument("--sources", type=Path, default=Path("data/snapshots/version_change_cohort_sources.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/holdout/version_change_cohort_manifest.json"))
    args = parser.parse_args()
    if args.sources.exists() or args.manifest.exists():
        parser.error("Frozen cohort exists; refusing to overwrite")
    listing_bytes = get_bytes(LISTING_URL)
    listing = json.loads(listing_bytes)
    excluded = prior_ids()
    excluded.update(row["set_id"] for row in read_jsonl(Path("data/snapshots/version_holdout_sources.jsonl")))
    candidates = [e for e in listing["data"] if e.get("setid") not in excluded
                  and int(e.get("spl_version") or 0) >= 2]
    selected, screening = [], []
    for entry in candidates[:args.max_screen]:
        if len(selected) >= args.target:
            break
        set_id = entry["setid"]
        try:
            history_url = f"{BASE}/{set_id}/history.json"
            history = json.loads(get_bytes(history_url))["data"]["history"]
            pair = sorted(history, key=lambda v: int(v["spl_version"]), reverse=True)[:2]
            if len(pair) != 2 or int(pair[0]["spl_version"]) != int(pair[1]["spl_version"]) + 1:
                raise ValueError("nonconsecutive history")
            versions = [{"published_date": v["published_date"],
                         **get_version(set_id, int(v["spl_version"]))} for v in reversed(pair)]
            changes = changed_clinical_sections(*versions)
            screening.append({"set_id": set_id, "versions": [v["spl_version"] for v in versions],
                              "decision": "included" if changes else "no_clinical_section_text_change",
                              "clinical_change_sections": changes})
            if changes:
                selected.append({"set_id": set_id, "title": entry["title"],
                                 "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                 "versions": versions, "clinical_change_sections": changes,
                                 "annotation_status": "sealed_pending_human_review"})
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            screening.append({"set_id": set_id, "decision": "excluded_download_or_history",
                              "reason": str(exc)[:200]})
    args.sources.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.sources.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    manifest = {
        "status": "sealed_sources_no_model_exposure_pending_human_questions_and_adjudication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "listing_url": LISTING_URL, "listing_sha256": sha256(listing_bytes).hexdigest(),
        "sampling_rule": "After excluding all benchmark, development, and eight already frozen primary-holdout IDs, scan page-1 API order up to max_screen. Include first target consecutive version pairs whose new version changes exact text in a section with a predeclared clinical heading regex. Retain all negative screening decisions. This string rule does not establish a clinically material answer change.",
        "clinical_heading_regex": CLINICAL.pattern,
        "target_pairs": args.target, "max_screen": args.max_screen,
        "selected_pairs": len(selected), "screening": screening,
        "sources_path": str(args.sources),
        "sources_sha256": sha256(args.sources.read_bytes()).hexdigest(),
        "prohibitions": ["No model outputs, prompt tuning, or retrieval development before evaluation lock.",
                         "Clinical importance and answer changes require independent human judgment."],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "screened": len(screening),
                      "sources": str(args.sources), "manifest": str(args.manifest)}))


if __name__ == "__main__":
    main()
