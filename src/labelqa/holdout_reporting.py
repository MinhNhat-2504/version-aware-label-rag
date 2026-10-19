"""Shared, prespecified cohort and sample-flow accounting for holdout reports."""

from __future__ import annotations

from collections import defaultdict

from .data import portable_path
from .review_resolution import SILVER_MODE


COHORTS = {
    "version_holdout_sources.jsonl": "consecutive_pairs",
    "version_change_cohort_sources.jsonl": "text_change_enriched",
}
# The expansion uses the same two sampling rules, so its arms join the same strata.
SOURCE_STRATA = {
    **COHORTS,
    "version_expansion_consecutive_sources.jsonl": "consecutive_pairs",
    "version_expansion_change_sources.jsonl": "text_change_enriched",
}


def cohort_for(source_snapshot: str) -> str:
    cohort = SOURCE_STRATA.get(portable_path(source_snapshot).name)
    if cohort is None:
        raise ValueError(f"Unknown frozen source cohort: {source_snapshot}")
    return cohort


def sample_flow(gold: list[dict], manifest: dict) -> dict:
    """Account for each candidate version item without converting exclusions to failures."""
    exclusions = manifest.get("exclusions")
    if (not isinstance(exclusions, list) or manifest.get("accepted") != len(gold)
            or manifest.get("excluded") != len(exclusions)
            or manifest.get("candidate_items") != len(gold) + len(exclusions)):
        raise ValueError("Manual gold manifest has inconsistent candidate/accept/exclude counts")
    qids = [row["qid"] for row in gold] + [row["qid"] for row in exclusions]
    if len(qids) != len(set(qids)):
        raise ValueError("Candidate version item was accepted and excluded or duplicated")
    pairs: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    versions: dict[str, dict[int, dict[str, int]]] = defaultdict(dict)
    by_cohort = {}
    for cohort in COHORTS.values():
        by_cohort[cohort] = {"candidate_items": 0, "accepted_items": 0,
                             "excluded_items": 0, "answerable_accepted": 0,
                             "unanswerable_accepted": 0}
    for row in gold:
        cohort = cohort_for(row["source_snapshot"])
        pair = row["set_id"]
        version_counts = versions[pair].setdefault(row["spl_version"],
                                                   {"accepted": 0, "excluded": 0})
        version_counts["accepted"] += 1
        if pair in pairs and cohort not in pairs[pair]:
            raise ValueError(f"Label pair occurs in both source cohorts: {pair}")
        counts = pairs[pair].setdefault(cohort, {"accepted": 0, "excluded": 0})
        counts["accepted"] += 1
        by_cohort[cohort]["candidate_items"] += 1
        by_cohort[cohort]["accepted_items"] += 1
        by_cohort[cohort]["answerable_accepted" if row["answerable"]
                          else "unanswerable_accepted"] += 1
    for row in exclusions:
        has_resolution = (isinstance(row.get("resolution_mode"), str)
                          and isinstance(row.get("resolver_ids"), list)
                          and (bool(row["resolver_ids"])
                               or row["resolution_mode"] == SILVER_MODE))  # Rule-based, no signer.
        has_legacy_adjudicator = (isinstance(row.get("adjudicator_id"), str)
                                  and bool(row["adjudicator_id"].strip()))
        if (not isinstance(row.get("reason"), str) or not row["reason"].strip()
                or not (has_resolution or has_legacy_adjudicator)):
            raise ValueError(f"Excluded version item lacks a manual reason: {row['qid']}")
        cohort = cohort_for(row["source_snapshot"])
        pair = row["set_id"]
        version_counts = versions[pair].setdefault(row["spl_version"],
                                                   {"accepted": 0, "excluded": 0})
        version_counts["excluded"] += 1
        if pair in pairs and cohort not in pairs[pair]:
            raise ValueError(f"Label pair occurs in both source cohorts: {pair}")
        counts = pairs[pair].setdefault(cohort, {"accepted": 0, "excluded": 0})
        counts["excluded"] += 1
        by_cohort[cohort]["candidate_items"] += 1
        by_cohort[cohort]["excluded_items"] += 1
    for pair in pairs:
        if len(versions[pair]) != 2:
            raise ValueError(f"Candidate pair does not contain two SPL versions: {pair}")
    for cohort in COHORTS.values():
        selected = [pair for pair, value in pairs.items() if cohort in value]
        accepted_versions = [sum(version["accepted"] > 0 for version in versions[pair].values())
                             for pair in selected]
        by_cohort[cohort].update({
            "candidate_pairs": len(selected),
            "both_versions_accepted": accepted_versions.count(2),
            "one_version_accepted": accepted_versions.count(1),
            "both_versions_excluded": accepted_versions.count(0),
        })
    return {"candidate_items": len(qids), "candidate_pairs": len(pairs),
            "accepted_items": len(gold), "excluded_items": len(exclusions),
            "by_cohort": by_cohort,
            "exclusions": [{"qid": row["qid"], "set_id": row["set_id"],
                            "spl_version": row["spl_version"],
                            "cohort": cohort_for(row["source_snapshot"]),
                            "reason": row["reason"]} for row in exclusions]}
