"""Descriptive agreement between two source-review files; never a gold decision.

With one disclosed LLM annotator the numbers are expert-LLM agreement, not inter-human
reliability. Identical answer text and identical quotes are reported separately because
a high share can also signal that one file was copied from the other.
"""

from __future__ import annotations

from collections import Counter


def normalized(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def cohen_kappa(pairs: list[tuple]) -> float | None:
    """Unweighted Cohen's kappa; None when undefined (no items or no chance variation)."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(left == right for left, right in pairs) / n
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    expected = sum(left_counts[key] * right_counts[key] for key in left_counts) / (n * n)
    if expected == 1:
        return None
    return round((observed - expected) / (1 - expected), 4)


def categorical(pairs: list[tuple]) -> dict:
    return {"items": len(pairs), "agree": sum(left == right for left, right in pairs),
            "cohen_kappa": cohen_kappa(pairs),
            "cross_tab": dict(sorted(Counter(f"{left}/{right}" for left, right in pairs).items()))}


def quotes(review: dict) -> set[str]:
    return {normalized(span["quote"]) for spans in review.get("acceptable_evidence_spans_per_claim") or []
            for span in spans if isinstance(span, dict) and span.get("quote")}


def sections(review: dict) -> set[int]:
    return {index for group in review.get("acceptable_section_indices_per_claim") or [] for index in group}


def source_review_agreement(pairs: list[tuple[dict, dict]], reader_ids: list[str],
                            llm_ids: list[str]) -> dict:
    """Summarize decision, answerability, change judgment, answers and evidence overlap."""
    both_accept = [(a, b) for a, b in pairs
                   if a.get("item_decision") == "accept" and b.get("item_decision") == "accept"]
    both_answerable = [(a, b) for a, b in both_accept
                       if a.get("answerable_from_version") is True and b.get("answerable_from_version") is True]
    jaccard = []
    for a, b in both_answerable:
        union = sections(a) | sections(b)
        jaccard.append(len(sections(a) & sections(b)) / len(union) if union else 1.0)
    quote_pairs = [(quotes(a), quotes(b)) for a, b in both_answerable]
    first_quotes = sum(len(left) for left, _ in quote_pairs)
    second_quotes = sum(len(right) for _, right in quote_pairs)
    shared = sum(len(left & right) for left, right in quote_pairs)
    return {
        "kind": "expert_llm_agreement_not_inter_human_reliability" if llm_ids else "inter_reader_agreement",
        "reader_ids": list(reader_ids), "llm_annotator_ids": list(llm_ids), "items": len(pairs),
        "item_decision": categorical([(a.get("item_decision"), b.get("item_decision")) for a, b in pairs]),
        "answerable_from_version_when_both_accept": categorical(
            [(a.get("answerable_from_version"), b.get("answerable_from_version")) for a, b in both_accept]),
        "version_change_judgment_when_both_accept": categorical(
            [(a.get("version_change_judgment"), b.get("version_change_judgment")) for a, b in both_accept]),
        "identical_answer_text_when_both_accept": sum(
            normalized(a.get("independent_answer", "")) == normalized(b.get("independent_answer", ""))
            for a, b in both_accept),
        "identical_claim_lists_when_both_answerable": sum(
            [normalized(x) for x in a.get("required_claims", [])]
            == [normalized(x) for x in b.get("required_claims", [])] for a, b in both_answerable),
        "both_answerable_items": len(both_answerable),
        "mean_evidence_section_jaccard_when_both_answerable": (
            round(sum(jaccard) / len(jaccard), 4) if jaccard else None),
        "distinct_quotes": {"reader_a": first_quotes, "reader_b": second_quotes, "identical": shared},
        "interpretation": ("Descriptive only. Identical answer text or quotes may reflect the same "
                           "verbatim label wording or copying; gold is the signed manual resolution."),
    }
