"""Validate two independent review files and emit a manual adjudication queue.

When one file comes from a disclosed ``llm_`` annotator, the human file must carry a
blind receipt written by the single-reader self-check before the human saw any LLM
output, and the LLM file must match its provenance receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .data import portable_path, read_jsonl
from .review_agreement import source_review_agreement
from .review_resolution import is_llm


DECISIONS = {"accept", "rewrite", "reject"}
CHANGE_JUDGMENTS = {"material", "text_only", "stable", "uncertain"}
BLIND_RECEIPT_STATUS = "human_reader_file_frozen_before_any_llm_comparison"
LLM_PROVENANCE_STATUS = "disclosed_llm_annotation_not_human_review_not_gold"


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def blind_receipt(report: dict, reader: Path, basis: Path, basis_key: str = "candidates_sha256",
                  status: str = BLIND_RECEIPT_STATUS) -> dict:
    """Freeze a human file's hash and time before the LLM file or queue is shown."""
    reviewer = report["reader_ids"][0]
    if is_llm(reviewer):
        raise ValueError("LLM annotations use their provenance receipt, not a blind human receipt")
    return {"status": status, "reviewer_id": reviewer,
            "reader_file": reader.as_posix(), "reader_sha256": file_sha256(reader),
            basis_key: file_sha256(basis),
            "received_utc": datetime.now(timezone.utc).isoformat(), "self_check": report}


def verify_blind_receipts(reader_paths: dict[str, Path], basis: Path,
                          human_receipt: Path | None, llm_provenance: Path | None,
                          basis_key: str = "candidates_sha256", human_status: str = BLIND_RECEIPT_STATUS,
                          llm_status: str = LLM_PROVENANCE_STATUS) -> dict:
    """Check the human blind receipt and the LLM provenance against the exact reader bytes.

    ``basis`` is the frozen file both readers worked from: the candidate questions for
    source review, or the method-blind packet for answer review.
    """
    llm = [reader for reader in reader_paths if is_llm(reader)]
    human = [reader for reader in reader_paths if not is_llm(reader)]
    if len(llm) != 1 or len(human) != 1:
        raise ValueError("Blind receipts apply to one human reader and one LLM annotator")
    if human_receipt is None or llm_provenance is None:
        raise ValueError("An LLM annotation needs the human blind receipt and the LLM provenance receipt")
    receipt = json.loads(human_receipt.read_text(encoding="utf-8"))
    if (receipt.get("status") != human_status or receipt.get("reviewer_id") != human[0]
            or receipt.get("reader_sha256") != file_sha256(reader_paths[human[0]])
            or receipt.get(basis_key) != file_sha256(basis)):
        raise ValueError("Human reader file differs from its blind receipt; it changed after freezing")
    provenance = json.loads(llm_provenance.read_text(encoding="utf-8"))
    if (provenance.get("status") != llm_status or provenance.get("annotator_id") != llm[0]
            or provenance.get("annotation_sha256") != file_sha256(reader_paths[llm[0]])
            or provenance.get(basis_key) != file_sha256(basis)):
        raise ValueError("LLM annotation differs from its provenance receipt")
    return {"human_reader": {"receipt": human_receipt.as_posix(),
                             "receipt_sha256": file_sha256(human_receipt),
                             "reader_sha256": receipt["reader_sha256"],
                             "received_utc": receipt["received_utc"]},
            "llm_annotation": {"provenance": llm_provenance.as_posix(),
                               "provenance_sha256": file_sha256(llm_provenance),
                               "annotation_sha256": provenance["annotation_sha256"]}}


def normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def span_in_passage(span: dict, passage) -> bool:
    """The post-gold mapping rule: same section and kind, exact row, quote inside one passage."""
    if passage.section_index != span["section_index"] or passage.kind != span["source_kind"]:
        return False
    if passage.kind == "table_row" and (passage.table_index != span.get("table_index")
                                         or passage.row_index != span.get("row_index")):
        return False
    return normalized(span["quote"]) in normalized(passage.text)


def validate_span_locators(qid: str, span_groups: list[list[dict]], passages: list) -> None:
    """Fail now if a reader span could not be mapped to a parsed passage after gold."""
    for claim_index, spans in enumerate(span_groups, 1):
        for span in spans:
            if not any(span_in_passage(span, passage) for passage in passages):
                where = (f"table {span.get('table_index')}, row {span.get('row_index')}"
                         if span["source_kind"] == "table_row" else "prose window")
                raise ValueError(f"Claim {claim_index} quote is not inside one parsed {where} "
                                 f"of section {span['section_index']} for {qid}")


def xml_passages(xml_dir: Path, set_id: str, version: int, xml_sha256: str, cache: dict) -> list:
    key = (set_id, version)
    if key not in cache:
        from hashlib import sha256
        from .spl_tables import parse_structured_passages
        xml_bytes = (xml_dir / f"{set_id}_v{version}.xml").read_bytes()
        if sha256(xml_bytes).hexdigest() != xml_sha256:
            raise ValueError(f"Review XML checksum mismatch for {set_id} v{version}")
        cache[key] = parse_structured_passages(xml_bytes)
    return cache[key]


def validate_evidence_spans(qid: str, claims: list[str], section_groups: list[list[int]],
                            span_groups: list[list[dict]], sections: dict[int, str]) -> None:
    if len(span_groups) != len(claims):
        raise ValueError(f"Missing claim-level source spans for {qid}")
    for indices, spans in zip(section_groups, span_groups):
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"Missing acceptable source span for {qid}")
        used = set()
        for span in spans:
            if not isinstance(span, dict):
                raise ValueError(f"Invalid source span for {qid}")
            section = span.get("section_index")
            quote = span.get("quote")
            kind = span.get("source_kind")
            if type(section) is not int or section not in sections or kind not in ("prose", "table_row"):
                raise ValueError(f"Invalid source span location for {qid}")
            if not isinstance(quote, str) or not quote.strip() or normalized(quote) not in normalized(sections[section]):
                raise ValueError(f"Source quote absent from pinned section for {qid}")
            if kind == "table_row":
                if (type(span.get("table_index")) is not int or span["table_index"] < 0
                        or type(span.get("row_index")) is not int or span["row_index"] < 0):
                    raise ValueError(f"Table row locator missing for {qid}")
            elif span.get("table_index") is not None or span.get("row_index") is not None:
                raise ValueError(f"Prose source cannot have a table row locator for {qid}")
            used.add(section)
        if used != set(indices):
            raise ValueError(f"Source span sections differ from accepted section indices for {qid}")


def index_reviews(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    by_id = {row["qid"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate qid in {path}")
    return by_id


def check(candidates: Path, first: Path, second: Path | None, xml_dir: Path | None = None,
          human_receipt: Path | None = None, llm_provenance: Path | None = None) -> dict:
    """Validate two reader files and queue them; with no second file, self-check one reader."""
    goldless = index_reviews(candidates)
    a = index_reviews(first)
    b = index_reviews(second) if second is not None else a
    if set(a) != set(goldless) or set(b) != set(goldless):
        raise ValueError("Review qids must exactly match the candidate set")
    ids_a = {row.get("reviewer_id", "").strip() for row in a.values()}
    ids_b = {row.get("reviewer_id", "").strip() for row in b.values()}
    if second is None:
        if len(ids_a) != 1 or "" in ids_a:
            raise ValueError("One stable, nonempty reviewer ID is required")
    elif len(ids_a) != 1 or len(ids_b) != 1 or "" in ids_a | ids_b or ids_a == ids_b:
        raise ValueError("Two distinct, nonempty reviewer IDs are required")
    receipts = None
    if second is not None and any(is_llm(reader) for reader in ids_a | ids_b):
        receipts = verify_blind_receipts({next(iter(ids_a)): first, next(iter(ids_b)): second},
                                         candidates, human_receipt, llm_provenance)
    source_index = {}
    for source in {portable_path(row["source_snapshot"]) for row in goldless.values()}:
        for pair in read_jsonl(source):
            for version in pair["versions"]:
                source_index[(pair["set_id"], version["spl_version"])] = version
    queue = []
    counts = Counter()
    single_counts = Counter()
    passage_cache: dict = {}
    for qid, candidate in goldless.items():
        left, right = a[qid], b[qid]
        pinned = source_index.get((candidate["set_id"], candidate["spl_version"]))
        if pinned is None or pinned["xml_sha256"] != candidate["xml_sha256"]:
            raise ValueError(f"Pinned source identity/hash mismatch for {qid}")
        sections = {s["section_index"]: s["text"] for s in pinned["sections"]}
        valid_sections = set(sections)
        for review in ((left, right) if second is not None else (left,)):
            if any(review.get(key) != candidate.get(key)
                   for key in ("set_id", "spl_version", "xml_sha256", "question")):
                raise ValueError(f"Source identity/question mismatch for {qid}")
            if review.get("item_decision") not in DECISIONS:
                raise ValueError(f"Missing/invalid item decision for {qid}")
            if review["item_decision"] == "accept":
                if review.get("version_change_judgment") not in CHANGE_JUDGMENTS:
                    raise ValueError(f"Missing version-change judgment for {qid}")
                if review.get("answerable_from_version") not in (True, False):
                    raise ValueError(f"Missing answerability for {qid}")
                if not review.get("independent_answer", "").strip():
                    raise ValueError(f"Missing independent answer for {qid}")
                if review["answerable_from_version"] and (not review.get("required_claims")
                    or len(review["required_claims"]) != len(review.get("acceptable_section_indices_per_claim", []))):
                    raise ValueError(f"Incomplete claim/evidence mapping for {qid}")
                if review["answerable_from_version"]:
                    for claim, indices in zip(review["required_claims"], review["acceptable_section_indices_per_claim"]):
                        if not isinstance(claim, str) or not claim.strip() or not isinstance(indices, list) or not indices:
                            raise ValueError(f"Empty claim or evidence list for {qid}")
                        if any(type(index) is not int or index not in valid_sections for index in indices):
                            raise ValueError(f"Evidence section absent from pinned source for {qid}")
                    validate_evidence_spans(qid, review["required_claims"],
                                            review["acceptable_section_indices_per_claim"],
                                            review.get("acceptable_evidence_spans_per_claim", []), sections)
                    if xml_dir is not None:
                        validate_span_locators(qid, review["acceptable_evidence_spans_per_claim"],
                                               xml_passages(xml_dir, candidate["set_id"],
                                                            candidate["spl_version"],
                                                            candidate["xml_sha256"], passage_cache))
                elif (review.get("required_claims") or review.get("acceptable_section_indices_per_claim")
                      or review.get("acceptable_evidence_spans_per_claim")):
                    raise ValueError(f"Unanswerable item has positive evidence claims for {qid}")
        single_counts.update([left["item_decision"]])
        if second is None:
            continue
        status = "needs_adjudication"
        reasons = []
        if left["item_decision"] != right["item_decision"]:
            reasons.append("question_decision_disagreement")
        if left.get("answerable_from_version") != right.get("answerable_from_version"):
            reasons.append("answerability_disagreement")
        if left.get("independent_answer", "").strip().casefold() != right.get("independent_answer", "").strip().casefold():
            reasons.append("answer_text_differs_check_semantics")
        if (left.get("required_claims") != right.get("required_claims")
                or left.get("acceptable_section_indices_per_claim") != right.get("acceptable_section_indices_per_claim")
                or left.get("acceptable_evidence_spans_per_claim") != right.get("acceptable_evidence_spans_per_claim")):
            reasons.append("claim_or_evidence_differs_check_equivalence")
        if left.get("version_change_judgment") != right.get("version_change_judgment"):
            reasons.append("version_change_judgment_disagreement")
        if not reasons:
            reasons.append("confirm_matching_reviews_and_accept_or_reject_manually")
        counts.update([left["item_decision"] + "/" + right["item_decision"]])
        queue.append({"qid": qid, "set_id": candidate["set_id"], "spl_version": candidate["spl_version"],
                      "status": status, "reasons": reasons, "reader_a": left, "reader_b": right,
                      "resolution_mode": "", "consensus_reviewer_ids": [],
                      "expert_resolver_id": "", "adjudicator_id": "",
                      "final_decision": "", "final_answer": "",
                      "final_answerable_from_version": None,
                      "final_claims": [], "final_acceptable_section_indices_per_claim": [],
                      "final_acceptable_evidence_spans_per_claim": [],
                      "final_version_change_judgment": "",
                      "adjudication_notes": ""})
    if second is None:
        return {"status": "single_reader_self_check_passed_not_adjudication",
                "reader_ids": [next(iter(ids_a))], "items": len(a),
                "decisions": dict(single_counts), "locators_checked": xml_dir is not None}
    reader_ids = [next(iter(ids_a)), next(iter(ids_b))]
    report = {"reader_ids": reader_ids, "items": len(queue),
              "decision_pairs": dict(counts), "all_items_require_manual_adjudication": True,
              "agreement": source_review_agreement([(a[qid], b[qid]) for qid in goldless], reader_ids,
                                                   [reader for reader in reader_ids if is_llm(reader)])}
    if receipts is not None:
        report["blind_receipts"] = receipts
    report["queue"] = queue
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("data/holdout/version_question_candidates_expanded.jsonl"))
    parser.add_argument("--reader-a", type=Path, required=True)
    parser.add_argument("--reader-b", type=Path,
                        help="Omit to self-check one reader file; no queue is written")
    parser.add_argument("--output", type=Path, default=Path("data/review/version_holdout_adjudication_queue.json"))
    parser.add_argument("--xml-dir", type=Path, default=Path("results/pinned_xml_holdout_review"),
                        help="Pinned review XML used to verify prose-window and table-row locators")
    parser.add_argument("--skip-locator-check", action="store_true")
    parser.add_argument("--receipt", type=Path,
                        help="Self-check only: freeze a human file's hash and UTC time before any LLM comparison")
    parser.add_argument("--human-receipt", type=Path,
                        help="Blind receipt of the human file; required when the other file is an LLM annotation")
    parser.add_argument("--llm-provenance", type=Path,
                        help="Provenance receipt of the LLM annotation (from labelqa.build_llm_annotation)")
    args = parser.parse_args()
    if not args.skip_locator_check and not args.xml_dir.is_dir():
        parser.error(f"Review XML not found in {args.xml_dir}; run labelqa.fetch_review_xml first")
    if args.receipt is not None and (args.reader_b is not None or args.skip_locator_check):
        parser.error("--receipt needs a single reader file and the locator check")
    if args.receipt is not None and args.receipt.exists():
        parser.error("Blind receipt already exists; it is written once")
    report = check(args.candidates, args.reader_a, args.reader_b,
                   None if args.skip_locator_check else args.xml_dir,
                   args.human_receipt, args.llm_provenance)
    if args.reader_b is None:
        if args.receipt is not None:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt = blind_receipt(report, args.reader_a, args.candidates)
            args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report = {**report, "blind_receipt": str(args.receipt), "reader_sha256": receipt["reader_sha256"]}
        print(json.dumps(report, ensure_ascii=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": report["items"], "decision_pairs": report["decision_pairs"],
                      "agreement_kind": report["agreement"]["kind"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
