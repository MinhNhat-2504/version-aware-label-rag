"""Compile an AI silver standard from two disclosed AI source annotations; never human gold.

Rule (Analysis Plan Amendment 6), applied by code with no signer:

- an item enters the silver standard only if both AI annotations ``accept`` it and give
  the same answerability; every other item is excluded with its reason;
- the answer, atomic claims and exact evidence spans come from the Codex annotation;
- the version-change judgment is kept when both annotations agree, else ``uncertain``.

Both files must match their provenance receipts and pass the same schema and pinned-XML
locator checks as a human file. The output keeps the gold-file schema so retrieval and
mechanical scoring can run, but its manifest status names it an AI silver standard.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .check_holdout_reviews import LLM_PROVENANCE_STATUS, check, file_sha256
from .data import read_jsonl
from .review_agreement import source_review_agreement
from .review_resolution import SILVER_MODE, SILVER_STATUS, is_llm


CODEX_PROVENANCE_STATUS = "disclosed_codex_ai_simulation_not_human_review_not_gold"
DEFAULT_OUTPUT = Path("data/holdout/version_silver_ai.jsonl")
RULE = ("Include an item only when both AI annotations accept it with the same answerability; take the "
        "answer, claims and evidence spans from the Codex annotation; keep the change judgment when both "
        "agree, otherwise 'uncertain'. No human reviewed any item.")


def verify_provenance(llm: Path, llm_provenance: Path, codex: Path, codex_provenance: Path,
                      candidates: Path) -> None:
    candidates_hash = file_sha256(candidates)
    first = json.loads(llm_provenance.read_text(encoding="utf-8"))
    if (first.get("status") != LLM_PROVENANCE_STATUS or first.get("annotation_sha256") != file_sha256(llm)
            or first.get("candidates_sha256") != candidates_hash):
        raise ValueError("LLM annotation differs from its provenance receipt")
    second = json.loads(codex_provenance.read_text(encoding="utf-8"))
    if (second.get("status") != CODEX_PROVENANCE_STATUS or second.get("reader_sha256") != file_sha256(codex)
            or second.get("candidates_sha256") != candidates_hash):
        raise ValueError("Codex annotation differs from its provenance receipt")


def silver_rows(candidates: list[dict], llm: dict[str, dict], codex: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    accepted, excluded = [], []
    for candidate in candidates:
        qid = candidate["qid"]
        a, c = llm[qid], codex[qid]
        reason = None
        if a["item_decision"] != "accept" or c["item_decision"] != "accept":
            reason = f"not accepted by both AI annotations (llm={a['item_decision']}, codex={c['item_decision']})"
        elif a["answerable_from_version"] != c["answerable_from_version"]:
            reason = (f"AI annotations disagree on answerability (llm={a['answerable_from_version']}, "
                      f"codex={c['answerable_from_version']})")
        if reason:
            excluded.append({"qid": qid, "set_id": candidate["set_id"], "spl_version": candidate["spl_version"],
                             "source_snapshot": candidate["source_snapshot"], "reason": reason,
                             "resolution_mode": SILVER_MODE, "resolver_ids": [], "adjudicator_id": ""})
            continue
        agreed = a["version_change_judgment"] == c["version_change_judgment"]
        accepted.append({
            "qid": qid, "pair_id": candidate["pair_id"], "set_id": candidate["set_id"],
            "spl_version": candidate["spl_version"], "xml_sha256": candidate["xml_sha256"],
            "source_snapshot": candidate["source_snapshot"], "question": candidate["question"],
            "answerable": c["answerable_from_version"], "reference_answer": c["independent_answer"].strip(),
            "claims": [{"text": claim.strip(), "acceptable_section_indices": sections,
                        "acceptable_evidence_spans": spans}
                       for claim, sections, spans in zip(c["required_claims"],
                                                         c["acceptable_section_indices_per_claim"],
                                                         c["acceptable_evidence_spans_per_claim"])],
            "version_change_judgment": c["version_change_judgment"] if agreed else "uncertain",
            "adjudicator_id": "", "resolution_mode": SILVER_MODE, "resolver_ids": [],
            "adjudication_notes": ("AI silver rule: both AI annotations accepted with the same answerability; "
                                   "Codex claims; change judgment "
                                   + ("agreed." if agreed else
                                      f"differed (llm={a['version_change_judgment']}, "
                                      f"codex={c['version_change_judgment']}), so uncertain.")),
        })
    return accepted, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, default=Path("data/holdout/version_question_candidates_expanded.jsonl"))
    parser.add_argument("--llm-annotation", type=Path, default=Path("private_review/llm_annotator_claude.jsonl"))
    parser.add_argument("--llm-provenance", type=Path, default=Path("private_review/llm_annotator_claude.provenance.json"))
    parser.add_argument("--codex-annotation", type=Path, default=Path("data/review/codex_ai_expert_simulation.jsonl"))
    parser.add_argument("--codex-provenance", type=Path,
                        default=Path("data/review/codex_ai_expert_simulation.provenance.json"))
    parser.add_argument("--xml-dir", type=Path, default=Path("results/pinned_xml_holdout_review"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        parser.error("Silver standard already exists; choose a new versioned path")
    if not args.xml_dir.is_dir():
        parser.error(f"Pinned XML not found in {args.xml_dir}")
    verify_provenance(args.llm_annotation, args.llm_provenance, args.codex_annotation, args.codex_provenance,
                      args.candidates)
    reports = [check(args.candidates, path, None, args.xml_dir) for path in (args.llm_annotation, args.codex_annotation)]
    reader_ids = [report["reader_ids"][0] for report in reports]
    if not all(is_llm(reader) for reader in reader_ids) or len(set(reader_ids)) != 2:
        raise ValueError("The silver rule takes exactly two distinct disclosed AI annotations")
    candidates = list(read_jsonl(args.candidates))
    llm = {row["qid"]: row for row in read_jsonl(args.llm_annotation)}
    codex = {row["qid"]: row for row in read_jsonl(args.codex_annotation)}
    accepted, excluded = silver_rows(candidates, llm, codex)
    if not accepted:
        raise ValueError("No item satisfies the silver rule")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "status": SILVER_STATUS, "reference_kind": "ai_silver_standard_not_human_gold",
        "rule": RULE, "candidate_items": len(candidates), "accepted": len(accepted),
        "excluded": len(excluded), "exclusions": excluded,
        "change_judgment_uncertain": sum(row["version_change_judgment"] == "uncertain" for row in accepted),
        "reader_ids": reader_ids, "llm_annotator_ids": reader_ids,
        "resolution_mode": SILVER_MODE, "resolver_ids": [],
        "reader_agreement": source_review_agreement([(llm[c["qid"]], codex[c["qid"]]) for c in candidates],
                                                    reader_ids, reader_ids),
        "disclosures": ["No human reviewed any item; this is an AI silver standard, not gold.",
                        "The Codex annotation was made after Codex had audited the LLM annotation, so the two "
                        "AI annotations are not independent and their agreement is inflated.",
                        "The same assistant drafted the expansion questions, built the pipeline and wrote the "
                        "LLM annotation."],
        "source_sha256": {path.as_posix(): file_sha256(path) for path in
                          (args.candidates, args.llm_annotation, args.codex_annotation, args.codex_provenance)},
        "provenance_sha256": {args.llm_provenance.as_posix(): file_sha256(args.llm_provenance)},
        "gold_sha256": sha256(args.output.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), "accepted": len(accepted),
                      "excluded": len(excluded), "change_judgment_uncertain": manifest["change_judgment_uncertain"]}))


if __name__ == "__main__":
    main()
