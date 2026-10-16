"""Draft version-paired questions and blank, independent reader packets."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


SOURCES = (
    Path("data/snapshots/version_holdout_sources.jsonl"),
    Path("data/snapshots/version_change_cohort_sources.jsonl"),
)
QUESTIONS = {
    "384cb547-55a4-47fc-a0a6-054c18a51b84": "Besides renal transplant rejection prophylaxis, what condition is sirolimus oral solution indicated to treat?",
    "3bad23a6-09a6-4194-9182-093ed61bc71c": "For which patients does the CANCIDAS label indicate treatment of invasive aspergillosis?",
    "3bea11a3-eb07-46b2-9b91-f8a48a2f3b7e": "What condition is lisinopril and hydrochlorothiazide indicated to treat?",
    "4bb241af-4299-4373-8762-2d6709515db0": "What does this DARZALEX FASPRO label say about its indication for high-risk smoldering multiple myeloma?",
    "4cab600d-6950-409f-a6d8-d447da22cd95": "What rheumatoid arthritis population is YUSIMRY indicated to treat?",
    "4deea61e-f3ca-b35d-e063-6294a90a9d41": "What does the label say about ciprofloxacin for skin and skin structure infections?",
    "4f6ae43f-c28a-4eac-9b19-8b5d0c12dfaa": "Which arrhythmias does this verapamil injection label list as indications?",
    "5701a07f-bd22-4bc7-95ff-0c5444ea5971": "What is the stated indication for triamcinolone acetonide cream?",
    "611f4a03-5b6f-420a-925e-adb905bf7b51": "What activity concentration range and vial volume does this version list for Ammonia N 13 Injection?",
    "83367367-5654-4078-831a-69ade48bc795": "What does the indication section of this tramadol extended-release label say about immediate-release opioids as an alternative?",
    "84b12129-cf10-4a3e-889c-990ebabdece6": "Which tablet strengths and identifying marks does this tramadol label describe?",
    "88156c4e-bc29-497d-8110-e59d800ff38a": "Which numbered warning does the Pediatric Use section of this topiramate label cite for pediatric epilepsy?",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    candidates = []
    for source in SOURCES:
        for pair in read_jsonl(source):
            set_id = pair["set_id"]
            if set_id not in QUESTIONS:
                raise ValueError(f"Question missing for {set_id}")
            for version in pair["versions"]:
                candidates.append({
                    "qid": f"vh-{set_id[:8]}-v{version['spl_version']}",
                    "pair_id": set_id, "set_id": set_id,
                    "spl_version": version["spl_version"],
                    "published_date": version["published_date"],
                    "xml_sha256": version["xml_sha256"],
                    "source_url": version["source_url"],
                    "source_snapshot": str(source),
                    "question": QUESTIONS[set_id],
                    "question_origin": "assistant_draft_before_any_model_holdout_run",
                    "annotation_status": "candidate_pending_two_independent_readers",
                })
    candidate_path = Path("data/holdout/version_question_candidates.jsonl")
    if candidate_path.exists():
        raise FileExistsError(candidate_path)
    write_jsonl(candidate_path, candidates)
    for reader in ("a", "b"):
        packet = []
        for row in candidates:
            packet.append({
                **row, "reviewer_id": "", "item_decision": "",
                "independent_answer": "", "answerable_from_version": None,
                "required_claims": [], "acceptable_section_indices_per_claim": [],
                "version_change_judgment": "",  # material, text_only, stable, uncertain
                "clinical_significance": "", "notes": "",
            })
        write_jsonl(Path(f"data/review/version_holdout_reader_{reader}_blank.jsonl"), packet)
    manifest_path = Path("data/holdout/version_test_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["question_candidates_path"] = str(candidate_path)
    manifest["question_candidates_sha256"] = sha256(candidate_path.read_bytes()).hexdigest()
    manifest["question_candidate_count"] = len(candidates)
    manifest["question_status"] = "assistant_drafts_pending_two_independent_readers; no gold answers yet"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(candidates) // 2, "candidate_questions": len(candidates),
                      "review_packets": 2}))


if __name__ == "__main__":
    main()
