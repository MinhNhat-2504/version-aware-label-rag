"""Exact equivalence plus optional question-specific adjudicated alternates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata

from .data import load_labels, load_questions, portable_path, split_label, validate_data
from .check_equivalence_reviews import frozen_candidates
from .retrieval import LabelIndex, tokenize
from .review_resolution import accepted_statuses, validate_compiled_resolution


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def passage_groups(chunks: tuple[str, ...]) -> dict[int, frozenset[int]]:
    groups: dict[str, set[int]] = defaultdict(set)
    for index, chunk in enumerate(chunks):
        groups[normalized_text(chunk)].add(index)
    return {index: frozenset(groups[normalized_text(chunk)]) for index, chunk in enumerate(chunks)}


def complete_with_equivalents(gold: tuple[int, ...], retrieved: list[int], groups: dict[int, frozenset[int]]) -> bool:
    found = set(retrieved)
    return all(found & groups[index] for index in gold)


def near_candidates(chunks: tuple[str, ...], gold_index: int, groups: dict[int, frozenset[int]], n: int = 3) -> list[dict]:
    gold_tokens = set(tokenize(chunks[gold_index]))
    if len(gold_tokens) < 12:
        return []
    candidates = []
    for index, chunk in enumerate(chunks):
        if index in groups[gold_index]:
            continue
        tokens = set(tokenize(chunk))
        if len(tokens) < 12:
            continue
        overlap = len(gold_tokens & tokens) / len(gold_tokens | tokens)
        if overlap >= 0.40:
            candidates.append({
                "index": index,
                "token_jaccard": round(overlap, 3),
                "gold_numbers": re.findall(r"\b\d+(?:\.\d+)?\b", chunks[gold_index]),
                "candidate_numbers": re.findall(r"\b\d+(?:\.\d+)?\b", chunk),
            })
    return sorted(candidates, key=lambda item: (-item["token_jaccard"], item["index"]))[:n]


def load_adjudications(path: Path) -> tuple[dict, str, dict]:
    receipt_path = path.with_suffix(".manifest.json")
    status = "assistant provisional, not expert or independent human validation"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("status") not in accepted_statuses("equivalence") or
                receipt.get("final_sha256") != sha256(path.read_bytes()).hexdigest()):
            raise ValueError("Invalid compiled equivalence receipt")
        sources = receipt.get("source_sha256")
        if not isinstance(sources, dict) or len(sources) != 6:
            raise ValueError("Compiled equivalence receipt lacks independent source files")
        for source, digest in sources.items():
            if sha256(portable_path(source).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Equivalence review source receipt changed: {source}")
        frozen_manifest = Path(receipt["frozen_manifest"])
        blank_a, blank_b = Path(receipt["blank_a"]), Path(receipt["blank_b"])
        if any(str(source) not in sources for source in (frozen_manifest, blank_a, blank_b)):
            raise ValueError("Compiled equivalence receipt lacks frozen candidate paths")
        blank = frozen_candidates(frozen_manifest, blank_a, blank_b)
        readers = receipt.get("reader_ids")
        if not isinstance(readers, list) or len(readers) != 2 or len(set(readers)) != 2:
            raise ValueError("Invalid independent equivalence reviewer identities")
        status = {"two_reader_consensus": "independently_reviewed_two_reader_consensus_development_validation_subset",
                  "human_expert_resolution": "human_expert_resolution_of_llm_annotation_development_validation_subset",
                  }.get(receipt.get("resolution_mode"), "independently_adjudicated_development_validation_subset")
    adjudications = {}
    counts = {"accept": 0, "reject": 0, "uncertain": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["qid"], row["gold_index"], row["candidate_index"])
        allowed = {"accept", "reject", "uncertain"} if receipt_path.is_file() else {"accept", "reject"}
        if key in adjudications or row["decision"] not in allowed:
            raise ValueError(f"Duplicate or invalid adjudication: {key}")
        adjudications[key] = row
        counts[row["decision"]] += 1
    if receipt_path.is_file():
        if len(adjudications) != receipt.get("items") or counts != receipt.get("decision_counts"):
            raise ValueError("Compiled equivalence decisions differ from receipt")
        by_review_id = {row.get("review_id"): row for row in adjudications.values()}
        if set(by_review_id) != set(blank):
            raise ValueError("Compiled equivalence candidate IDs differ from frozen packet")
        for rid, row in by_review_id.items():
            validate_compiled_resolution(row, receipt, readers, rid)
            source = blank[rid]
            if (row["qid"] != source["qid"] or row["set_id"] != source["set_id"] or
                    row["gold_index"] != source["original_index"] or
                    row["candidate_index"] != source["candidate_index"] or
                    not isinstance(row.get("adjudication_notes"), str) or
                    not row["adjudication_notes"].strip()):
                raise ValueError(f"Compiled equivalence row differs from frozen candidate: {rid}")
    return adjudications, status, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/upstream/FDARxBench/data"))
    parser.add_argument("--split", choices=["dev", "validation"], default="validation")
    parser.add_argument("--report", type=Path, default=Path("results/equivalence_validation.json"))
    parser.add_argument("--review", type=Path, default=Path("results/equivalence_review_validation.jsonl"))
    parser.add_argument("--review-limit", type=int, default=100)
    parser.add_argument("--adjudications", type=Path, default=None,
                        help="Question-specific alternate passage decisions; compiled human files need a receipt")
    args = parser.parse_args()
    frozen_manifest = Path("data/review/equivalence_validation_manifest.json")
    if frozen_manifest.is_file():
        frozen_source = Path(json.loads(frozen_manifest.read_text(encoding="utf-8"))["source"])
        if args.review.resolve() == frozen_source.resolve():
            raise ValueError("Refusing to overwrite the frozen equivalence-review source; use a new --review path")
    labels = load_labels(args.source / "labels" / "labels.jsonl")
    questions = load_questions(args.source / "qa" / "qa.jsonl")
    validate_data(labels, questions)
    indexes = {}
    groups_by_label = {}
    adjudications = {}
    adjudication_status = "none"
    adjudication_decisions = {"accept": 0, "reject": 0, "uncertain": 0}
    if args.adjudications:
        adjudications, adjudication_status, adjudication_decisions = load_adjudications(args.adjudications)
        receipt_path = args.adjudications.with_suffix(".manifest.json")
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            protected = {portable_path(source).resolve() for source in receipt["source_sha256"]}
            if args.review.resolve() in protected or args.report.resolve() in protected:
                raise ValueError("Output path would overwrite a frozen human-review source")
    review_metric = ("partial_independent_review_complete" if
                     adjudication_status in {
                         "independently_adjudicated_development_validation_subset",
                         "independently_reviewed_two_reader_consensus_development_validation_subset",
                         "human_expert_resolution_of_llm_annotation_development_validation_subset"}
                     else "provisional_review_complete")
    counts = {method: {"eligible": 0, "literal_complete": 0, "exact_equiv_complete": 0,
                       review_metric: 0} for method in ("bm25", "ontology_cover")}
    reviewed_qids = {qid for qid, _, _ in adjudications}
    reviewed_subset = {method: {"eligible": 0, "literal_complete": 0,
                                "exact_equiv_complete": 0, review_metric: 0}
                       for method in counts}
    duplicate_gold_questions = 0
    review_rows = []
    review_labels = set()
    for q in questions:
        if q.task != "multihop" or split_label(q.set_id) != args.split:
            continue
        label = labels[q.set_id]
        if q.set_id not in indexes:
            indexes[q.set_id] = LabelIndex(label)
            groups_by_label[q.set_id] = passage_groups(label.chunks)
        groups = groups_by_label[q.set_id]
        if len({groups[index] for index in q.gold}) < len(q.gold):
            duplicate_gold_questions += 1
        if len(q.gold) > 5:
            continue
        for method in counts:
            ranking = indexes[q.set_id].retrieve(q.text, method)
            reviewed_groups = {
                index: groups[index] | frozenset(
                    candidate for (qid, gold, candidate), row in adjudications.items()
                    if qid == q.qid and gold == index and row["decision"] == "accept"
                ) for index in q.gold
            }
            literal = int(set(q.gold) <= set(ranking))
            exact = int(complete_with_equivalents(q.gold, ranking, groups))
            with_review = int(all(set(ranking) & reviewed_groups[index] for index in q.gold))
            buckets = [counts[method]]
            if q.qid in reviewed_qids:
                buckets.append(reviewed_subset[method])
            for bucket in buckets:
                bucket["eligible"] += 1
                bucket["literal_complete"] += literal
                bucket["exact_equiv_complete"] += exact
                bucket[review_metric] += with_review
        if len(review_rows) < args.review_limit and q.set_id not in review_labels:
            for gold_index in q.gold:
                near = near_candidates(label.chunks, gold_index, groups)
                if near:
                    review_rows.append({
                        "qid": q.qid,
                        "set_id": q.set_id,
                        "question": q.text,
                        "reference_answer": q.answer,
                        "gold_index": gold_index,
                        "gold_text": label.chunks[gold_index],
                        "exact_equivalent_indices": sorted(groups[gold_index]),
                        "near_candidates_unverified": [
                            {**candidate, "text": label.chunks[candidate["index"]]}
                            for candidate in near
                        ],
                        "human_adjudication": {"accepted_indices": [], "rejected_indices": [], "notes": ""},
                    })
                    review_labels.add(q.set_id)
                    if len(review_rows) >= args.review_limit:
                        break
    report = {
        "split": args.split,
        "method": "Unicode NFKC + casefold + whitespace normalization; only exact matches are accepted automatically",
        "duplicate_gold_questions": duplicate_gold_questions,
        "counts": counts,
        "rates": {
            method: {
                name: round(value / values["eligible"], 4)
                for name, value in values.items() if name != "eligible"
            } for method, values in counts.items()
        },
        "near_candidate_review_rows": len(review_rows),
        "adjudication_count": len(adjudications),
        "adjudication_status": adjudication_status,
        "adjudication_decisions": adjudication_decisions,
        "review_coverage": {"candidate_pairs_decided": len(adjudications),
                            "questions_with_candidate_decision": len(reviewed_qids),
                            "eligible_questions_in_split": counts["bm25"]["eligible"],
                            "note": "Candidate pairs are selected, not a random or exhaustive review of all alternate evidence."},
        "reviewed_question_subset_counts": reviewed_subset,
        "reviewed_question_subset_rates": {
            method: {
                name: round(value / values["eligible"], 4)
                for name, value in values.items() if name != "eligible"
            } if values["eligible"] else {}
            for method, values in reviewed_subset.items()
        },
        "review_sha256": None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.review.parent.mkdir(parents=True, exist_ok=True)
    with args.review.open("w", encoding="utf-8") as handle:
        for row in review_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report["review_sha256"] = sha256(args.review.read_bytes()).hexdigest()
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
