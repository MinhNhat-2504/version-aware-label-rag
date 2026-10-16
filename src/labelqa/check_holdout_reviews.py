"""Validate two independent review files and emit a manual adjudication queue."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .data import portable_path, read_jsonl


DECISIONS = {"accept", "rewrite", "reject"}
CHANGE_JUDGMENTS = {"material", "text_only", "stable", "uncertain"}


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


def check(candidates: Path, first: Path, second: Path | None, xml_dir: Path | None = None) -> dict:
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
    return {"reader_ids": [next(iter(ids_a)), next(iter(ids_b))], "items": len(queue),
            "decision_pairs": dict(counts), "all_items_require_manual_adjudication": True,
            "queue": queue}


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
    args = parser.parse_args()
    if not args.skip_locator_check and not args.xml_dir.is_dir():
        parser.error(f"Review XML not found in {args.xml_dir}; run labelqa.fetch_review_xml first")
    report = check(args.candidates, args.reader_a, args.reader_b,
                   None if args.skip_locator_check else args.xml_dir)
    if args.reader_b is None:
        print(json.dumps(report, ensure_ascii=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"items": report["items"], "decision_pairs": report["decision_pairs"],
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
