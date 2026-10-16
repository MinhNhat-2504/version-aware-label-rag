"""Convert a disclosed LLM annotator's private notes into a checked reader file.

The private specification records, per candidate item, the annotator's decision,
answerability, answer, atomic claims and verbatim evidence quotes. This tool resolves
each quote to the parser's exact prose-window or table-row locator, writes the reader
file under an ``llm_`` reviewer ID, runs the same single-reader check used for humans,
and writes a provenance receipt. The output is one independent annotation; it is never
gold until a human expert resolves every item.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .check_holdout_reviews import check, normalized, xml_passages
from .data import read_jsonl
from .review_resolution import is_llm


DEFAULT_CANDIDATES = Path("data/holdout/version_question_candidates_expanded.jsonl")
DEFAULT_XML_DIR = Path("results/pinned_xml_holdout_review")
DECISIONS = {"accept", "rewrite", "reject"}
CHANGES = {"material", "text_only", "stable", "uncertain"}


def locate(span: dict, passages: list, qid: str) -> dict:
    """Resolve one quote; table rows must be unique unless coordinates are given."""
    kind = span.get("kind", "prose")
    quote = span["quote"]
    matches = [p for p in passages if p.section_index == span["section_index"] and p.kind == kind
               and normalized(quote) in normalized(p.text)]
    if kind == "table_row":
        if span.get("table_index") is not None:
            matches = [p for p in matches if (p.table_index, p.row_index)
                       == (span["table_index"], span["row_index"])]
        if len({(p.table_index, p.row_index) for p in matches}) != 1:
            raise ValueError(f"{qid}: table quote must match exactly one row: {quote[:60]}")
        return {"section_index": span["section_index"], "source_kind": kind, "quote": quote,
                "table_index": matches[0].table_index, "row_index": matches[0].row_index}
    if not matches:
        raise ValueError(f"{qid}: prose quote is not inside one parsed window: {quote[:60]}")
    return {"section_index": span["section_index"], "source_kind": kind, "quote": quote,
            "table_index": None, "row_index": None}


def build_rows(spec: dict, candidates: list[dict], xml_dir: Path) -> list[dict]:
    annotator = spec["annotator_id"]
    if not is_llm(annotator):
        raise ValueError("LLM annotations must use an llm_ reviewer ID")
    items = spec["items"]
    if set(items) != {row["qid"] for row in candidates}:
        raise ValueError("Annotation must cover exactly the candidate items")
    cache: dict = {}
    rows = []
    for candidate in candidates:
        qid, item = candidate["qid"], items[candidate["qid"]]
        if item["decision"] not in DECISIONS:
            raise ValueError(f"{qid}: invalid decision")
        row = {**candidate, "reviewer_id": annotator, "item_decision": item["decision"],
               "independent_answer": "", "answerable_from_version": None, "required_claims": [],
               "acceptable_section_indices_per_claim": [], "version_change_judgment": "",
               "clinical_significance": "", "notes": item.get("notes", ""),
               "acceptable_evidence_spans_per_claim": []}
        if item["decision"] == "accept":
            if item["version_change"] not in CHANGES or type(item["answerable"]) is not bool:
                raise ValueError(f"{qid}: missing change judgment or answerability")
            row.update({"independent_answer": item["answer"].strip(),
                        "answerable_from_version": item["answerable"],
                        "version_change_judgment": item["version_change"]})
            if item["answerable"]:
                passages = xml_passages(xml_dir, candidate["set_id"], candidate["spl_version"],
                                        candidate["xml_sha256"], cache)
                spans = [[locate(span, passages, qid) for span in claim["evidence"]]
                         for claim in item["claims"]]
                row.update({"required_claims": [claim["text"].strip() for claim in item["claims"]],
                            "acceptable_evidence_spans_per_claim": spans,
                            "acceptable_section_indices_per_claim": [
                                sorted({span["section_index"] for span in group}) for group in spans]})
            elif item.get("claims"):
                raise ValueError(f"{qid}: unanswerable item cannot carry claims")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt_path = args.output.with_suffix(".provenance.json")
    if args.output.exists() or receipt_path.exists():
        parser.error("Annotation output exists; write a new versioned file")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    candidates = list(read_jsonl(args.candidates))
    rows = build_rows(spec, candidates, args.xml_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
    report = check(args.candidates, args.output, None, args.xml_dir)
    receipt = {
        "status": "disclosed_llm_annotation_not_human_review_not_gold",
        "annotator_id": spec["annotator_id"], "model": spec["model"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disclosures": spec["disclosures"],
        "spec_sha256": sha256(args.spec.read_bytes()).hexdigest(),
        "candidates_sha256": sha256(args.candidates.read_bytes()).hexdigest(),
        "annotation_sha256": sha256(args.output.read_bytes()).hexdigest(),
        "self_check": report,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "items": len(rows), "self_check": report["status"]}))


if __name__ == "__main__":
    main()
