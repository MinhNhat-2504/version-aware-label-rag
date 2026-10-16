"""Freeze a larger version-pair expansion cohort before any human gold or model output exists."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable, Iterable, Iterator
from urllib.parse import urlencode
import xml.etree.ElementTree as ET
from zipfile import BadZipFile

from .build_change_cohort import CLINICAL, changed_clinical_sections
from .build_version_holdout import prior_ids
from .data import read_jsonl
from .recent import BASE, get_bytes
from .version_pairs import get_version


EXISTING_COHORTS = (
    Path("data/snapshots/version_holdout_sources.jsonl"),
    Path("data/snapshots/version_change_cohort_sources.jsonl"),
)
CONSECUTIVE_OUTPUT = Path("data/snapshots/version_expansion_consecutive_sources.jsonl")
CHANGE_OUTPUT = Path("data/snapshots/version_expansion_change_sources.jsonl")
MANIFEST_OUTPUT = Path("data/holdout/version_expansion_manifest.json")
BLANK_PACKETS = (
    Path("data/review/version_holdout_reader_a_blank.jsonl"),
    Path("data/review/version_holdout_reader_b_blank.jsonl"),
)
EXPOSURE_ARTIFACTS = (
    Path("data/holdout/version_gold_adjudicated.jsonl"),
    Path("data/holdout/version_gold_adjudicated.manifest.json"),
    Path("data/review/version_holdout_adjudication_queue.json"),
    Path("paper/HOLDOUT_EXECUTION_LOCK.json"),
)
PREFLIGHT = Path("results/holdout_source_preflight.json")
LISTING_PARAMS = {"doctype": "34391-3", "published_date": "2026-09-03",
                  "published_date_comparison": "gte", "pagesize": 100}
TARGET_CONSECUTIVE = 32
TARGET_CHANGED = 16
MAX_CHANGE_SCREEN = 150
PAUSE_SECONDS = 0.2
FETCH_ERRORS = (OSError, ValueError, KeyError, json.JSONDecodeError, ET.ParseError, BadZipFile)


def pre_exposure_checks(root: Path = Path(".")) -> dict:
    """Refuse expansion once any human gold, review queue, lock or holdout output exists."""
    present = [str(path) for path in EXPOSURE_ARTIFACTS if (root / path).exists()]
    present += sorted(str(path.relative_to(root)) for path in (root / "results").glob("holdout_*_prompts*"))
    present += sorted(str(path.relative_to(root)) for path in (root / "results").glob("holdout_*_responses*"))
    if present:
        raise ValueError("Expansion must precede human gold and model exposure; found: " + ", ".join(present))
    preflight = json.loads((root / PREFLIGHT).read_text(encoding="utf-8"))
    if (preflight.get("independent_human_adjudications") != 0
            or preflight.get("holdout_model_evaluations") != 0):
        raise ValueError("Frozen preflight records human adjudication or model evaluation")
    for packet in BLANK_PACKETS:
        for row in read_jsonl(root / packet):
            if (row.get("reviewer_id") or row.get("item_decision") or row.get("independent_answer")
                    or row.get("required_claims") or row.get("acceptable_evidence_spans_per_claim")):
                raise ValueError(f"Reviewer packet already contains decisions: {packet}")
    return {"human_gold_or_review_queue_present": False, "execution_lock_present": False,
            "holdout_prompts_or_responses_present": False,
            "existing_blank_packets_unfilled": True,
            "preflight_human_adjudications": 0, "preflight_model_evaluations": 0}


def listing_url(page: int) -> str:
    return f"{BASE}.json?{urlencode({**LISTING_PARAMS, 'page': page})}"


def fetch_listing(page_log: list[dict], pause: float = PAUSE_SECONDS) -> Iterator[dict]:
    """Yield DailyMed listing entries in API order, logging each page checksum."""
    page = 1
    while True:
        url = listing_url(page)
        payload = get_bytes(url)
        listing = json.loads(payload)
        entries = listing.get("data", [])
        page_log.append({"page": page, "url": url, "sha256": sha256(payload).hexdigest(),
                         "entries": len(entries), "metadata": listing.get("metadata", {})})
        if not entries:
            return
        yield from entries
        total_pages = int(listing.get("metadata", {}).get("total_pages", page))
        if page >= total_pages:
            return
        page += 1
        time.sleep(pause)


def fetch_history(set_id: str) -> list[dict]:
    return json.loads(get_bytes(f"{BASE}/{set_id}/history.json"))["data"]["history"]


def freeze_expansion(entries: Iterable[dict], excluded: set[str],
                     history: Callable[[str], list[dict]],
                     version: Callable[[str, int], dict],
                     n_consecutive: int = TARGET_CONSECUTIVE, n_changed: int = TARGET_CHANGED,
                     max_change_screen: int = MAX_CHANGE_SCREEN,
                     now: Callable[[], str] = lambda: datetime.now(timezone.utc).isoformat(),
                     ) -> tuple[list[dict], list[dict], list[dict]]:
    """Fill the consecutive arm in listing order, then screen later labels for text changes."""
    if n_consecutive < 1 or n_changed < 1 or max_change_screen < n_changed:
        raise ValueError("Targets must be positive and the change screen must cover its target")
    consecutive: list[dict] = []
    changed: list[dict] = []
    screening: list[dict] = []
    seen: set[str] = set()
    change_screens = 0
    for entry in entries:
        if len(changed) >= n_changed or change_screens >= max_change_screen:
            break
        set_id = entry.get("setid")
        if (not set_id or set_id in excluded or set_id in seen
                or int(entry.get("spl_version") or 0) < 2):
            continue
        seen.add(set_id)
        arm = "consecutive" if len(consecutive) < n_consecutive else "change_screen"
        if arm == "change_screen":
            change_screens += 1
        try:
            ordered = sorted(history(set_id), key=lambda row: int(row["spl_version"]), reverse=True)
            if len(ordered) < 2 or int(ordered[0]["spl_version"]) != int(ordered[1]["spl_version"]) + 1:
                raise ValueError("latest two history versions are not consecutive")
            versions = [{"published_date": meta["published_date"],
                         **version(set_id, int(meta["spl_version"]))}
                        for meta in reversed(ordered[:2])]
            if any(not item["sections"] for item in versions):
                raise ValueError("a pinned version has no parsed sections")
        except FETCH_ERRORS as exc:
            screening.append({"set_id": set_id, "arm": arm, "decision": "excluded_download_or_history",
                              "reason": str(exc)[:200]})
            continue
        changes = changed_clinical_sections(*versions)
        record = {"set_id": set_id, "title": entry.get("title"),
                  "listed_published_date": entry.get("published_date"),
                  "retrieved_utc": now(), "versions": versions,
                  "clinical_change_sections": changes,
                  "annotation_status": "sealed_pending_human_review"}
        if arm == "consecutive":
            decision = "included_consecutive"
            consecutive.append(record)
        elif changes:
            decision = "included_text_change"
            changed.append(record)
        else:
            decision = "no_clinical_section_text_change"
        screening.append({"set_id": set_id, "arm": arm, "decision": decision,
                          "versions": [item["spl_version"] for item in versions],
                          "clinical_change_sections": changes})
    return consecutive, changed, screening


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consecutive", type=int, default=TARGET_CONSECUTIVE)
    parser.add_argument("--changed", type=int, default=TARGET_CHANGED)
    parser.add_argument("--max-change-screen", type=int, default=MAX_CHANGE_SCREEN)
    args = parser.parse_args()
    outputs = (CONSECUTIVE_OUTPUT, CHANGE_OUTPUT, MANIFEST_OUTPUT)
    if any(path.exists() for path in outputs):
        parser.error("Expansion cohort already frozen; refusing to overwrite")
    checks = pre_exposure_checks()
    excluded = prior_ids()
    for path in EXISTING_COHORTS:
        excluded.update(row["set_id"] for row in read_jsonl(path))
    page_log: list[dict] = []

    def paced_history(set_id: str) -> list[dict]:
        time.sleep(PAUSE_SECONDS)
        return fetch_history(set_id)

    def paced_version(set_id: str, number: int) -> dict:
        time.sleep(PAUSE_SECONDS)
        return get_version(set_id, number)

    started = datetime.now(timezone.utc).isoformat()
    consecutive, changed, screening = freeze_expansion(
        fetch_listing(page_log), excluded, paced_history, paced_version,
        args.consecutive, args.changed, args.max_change_screen)
    if len(consecutive) != args.consecutive or len(changed) != args.changed:
        raise RuntimeError(f"Only {len(consecutive)}/{args.consecutive} consecutive and "
                           f"{len(changed)}/{args.changed} changed pairs; nothing was written")
    write_jsonl(CONSECUTIVE_OUTPUT, consecutive)
    write_jsonl(CHANGE_OUTPUT, changed)
    manifest = {
        "status": "sealed_expansion_sources_no_model_exposure_pending_human_questions_and_adjudication",
        "created_utc": datetime.now(timezone.utc).isoformat(), "screening_started_utc": started,
        "design_decision": (
            "Precision-oriented expansion approved by the project owner on 2026-10-17 after the "
            "metadata-only sensitivity analysis and before any human gold, review queue or holdout "
            "model output. It is not powered for the observed +2.33-point validation retrieval "
            "difference, which would need about 1,270 independent items for 80% power."),
        "listing_params": LISTING_PARAMS, "listing_pages": page_log,
        "sampling_rule": (
            "Paginate the DailyMed human-prescription listing in API order. Exclude all FDARxBench QA "
            "set IDs, every prior development snapshot and both already frozen holdout cohorts; require "
            "SPL version >= 2 and consecutive latest two history versions. Include the first "
            "consecutive_target eligible pairs regardless of content change. Then screen subsequent "
            "eligible labels and include the first changed_target pairs whose newer version changes "
            "exact text in a section matching the predeclared clinical-heading regex, up to "
            "max_change_screen screens. Retain every screening decision. Never replace a pair because "
            "of its content, question quality or any model result."),
        "clinical_heading_regex": CLINICAL.pattern,
        "targets": {"consecutive": args.consecutive, "changed": args.changed,
                    "max_change_screen": args.max_change_screen},
        "pre_exposure_checks": checks,
        "excluded_prior_set_ids": len(excluded),
        "screening": screening,
        "sources": {
            CONSECUTIVE_OUTPUT.as_posix(): {
                "arm": "consecutive_pairs", "selected_pairs": len(consecutive),
                "sha256": sha256(CONSECUTIVE_OUTPUT.read_bytes()).hexdigest()},
            CHANGE_OUTPUT.as_posix(): {
                "arm": "text_change_enriched", "selected_pairs": len(changed),
                "sha256": sha256(CHANGE_OUTPUT.read_bytes()).hexdigest()},
        },
        "prohibitions": [
            "No model outputs, prompt tuning or retrieval development on these sources before the evaluation lock.",
            "Assistant-drafted questions or judgments are not independent human annotations.",
            "A string change in a clinical heading does not establish a clinically material answer change.",
        ],
    }
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"consecutive": len(consecutive), "changed": len(changed),
                      "screened": len(screening), "listing_pages": len(page_log),
                      "manifest": str(MANIFEST_OUTPUT)}))


if __name__ == "__main__":
    main()
