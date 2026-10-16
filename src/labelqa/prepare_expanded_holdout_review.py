"""Combine the frozen 24 candidates with expansion questions into new blank reader packets."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .build_expansion_cohort import pre_exposure_checks
from .data import read_jsonl
from .expansion_questions import QUESTIONS
from .validate_holdout import EXPANSION_MANIFEST, frozen_receipt


ORIGINAL_CANDIDATES = Path("data/holdout/version_question_candidates.jsonl")
ORIGINAL_MANIFEST = Path("data/holdout/version_test_manifest.json")
EXPANSION_SOURCES = (
    Path("data/snapshots/version_expansion_consecutive_sources.jsonl"),
    Path("data/snapshots/version_expansion_change_sources.jsonl"),
)
CANDIDATES = Path("data/holdout/version_question_candidates_expanded.jsonl")
RECEIPT = Path("data/holdout/version_question_candidates_expanded.manifest.json")
PACKETS = {reader: Path(f"data/review/version_holdout_expanded_reader_{reader}_blank.jsonl")
           for reader in ("a", "b")}
BLANK_FIELDS = {
    "reviewer_id": "", "item_decision": "", "independent_answer": "",
    "answerable_from_version": None, "required_claims": [],
    "acceptable_section_indices_per_claim": [], "version_change_judgment": "",
    "clinical_significance": "", "notes": "", "acceptable_evidence_spans_per_claim": [],
}


def expansion_candidates(pairs_by_source: dict[Path, list[dict]],
                         questions: dict[str, str]) -> list[dict]:
    """One identical question per pair, applied to both pinned versions."""
    rows = []
    selected = {pair["set_id"] for pairs in pairs_by_source.values() for pair in pairs}
    if set(questions) != selected:
        raise ValueError("Expansion questions must cover exactly the frozen expansion pairs")
    for source, pairs in pairs_by_source.items():
        for pair in pairs:
            question = questions[pair["set_id"]].strip()
            if not question.endswith("?"):
                raise ValueError(f"Question draft is incomplete: {pair['set_id']}")
            for version in pair["versions"]:
                rows.append({
                    "qid": f"vx-{pair['set_id'][:8]}-v{version['spl_version']}",
                    "pair_id": pair["set_id"], "set_id": pair["set_id"],
                    "spl_version": version["spl_version"],
                    "published_date": version["published_date"],
                    "xml_sha256": version["xml_sha256"], "source_url": version["source_url"],
                    "source_snapshot": source.as_posix(), "question": question,
                    "question_origin": "assistant_draft_before_any_model_holdout_run",
                    "annotation_status": "candidate_pending_two_independent_readers",
                })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    outputs = (CANDIDATES, RECEIPT, *PACKETS.values())
    if any(path.exists() for path in outputs):
        raise FileExistsError("Expanded candidate files are frozen; write a new versioned file instead")
    checks = pre_exposure_checks()
    original_manifest = json.loads(ORIGINAL_MANIFEST.read_text(encoding="utf-8"))
    if sha256(ORIGINAL_CANDIDATES.read_bytes()).hexdigest() != original_manifest["question_candidates_sha256"]:
        raise ValueError("The original 24 frozen candidates changed")
    expansion_manifest = json.loads(EXPANSION_MANIFEST.read_text(encoding="utf-8"))
    pairs_by_source = {}
    for source in EXPANSION_SOURCES:
        expected_sha, expected_pairs = frozen_receipt(expansion_manifest, source)
        if sha256(source.read_bytes()).hexdigest() != expected_sha:
            raise ValueError(f"Expansion source checksum changed: {source}")
        pairs_by_source[source] = list(read_jsonl(source))
        if len(pairs_by_source[source]) != expected_pairs:
            raise ValueError(f"Expansion source count changed: {source}")
    original = list(read_jsonl(ORIGINAL_CANDIDATES))
    candidates = original + expansion_candidates(pairs_by_source, QUESTIONS)
    if len({row["qid"] for row in candidates}) != len(candidates):
        raise ValueError("Candidate question IDs collide")
    write_jsonl(CANDIDATES, candidates)
    for path in PACKETS.values():
        write_jsonl(path, [{**row, **BLANK_FIELDS} for row in candidates])
    receipt = {
        "status": "assistant_drafted_candidates_pending_two_independent_readers_no_gold",
        "original_candidates": ORIGINAL_CANDIDATES.as_posix(),
        "original_candidates_sha256": sha256(ORIGINAL_CANDIDATES.read_bytes()).hexdigest(),
        "expansion_manifest_sha256": sha256(EXPANSION_MANIFEST.read_bytes()).hexdigest(),
        "candidates": CANDIDATES.as_posix(),
        "candidates_sha256": sha256(CANDIDATES.read_bytes()).hexdigest(),
        "candidate_items": len(candidates),
        "label_pairs": len({row["pair_id"] for row in candidates}),
        "original_items_unchanged": len(original),
        "blank_packets_sha256": {path.as_posix(): sha256(path.read_bytes()).hexdigest()
                                 for path in PACKETS.values()},
        "pre_exposure_checks": checks,
        "note": ("Expansion questions were drafted by the assistant from frozen source text before "
                 "any retrieval or model call on these labels. They are candidates, not gold."),
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_items": len(candidates), "label_pairs": receipt["label_pairs"],
                      "packets": [str(path) for path in PACKETS.values()]}))


if __name__ == "__main__":
    main()
