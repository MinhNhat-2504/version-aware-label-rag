"""Mechanical citation and numeric-unit triage on frozen development responses."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re

from .data import read_jsonl
from .score_qa import validate_provenance


PASSAGE_MARKER = re.compile(r"(?m)^\[(P\d{4})\]\s*$")
NUMBER_UNIT = re.compile(
    r"(?<!\w)(\d+(?:\.\d+)?)\s*(mL/min|micrograms?|mcg|mg|g|mL|L|days?|years?|hours?|weeks?|months?|%)(?!\w)",
    re.I,
)
SINGULAR = {"days": "day", "years": "year", "hours": "hour", "weeks": "week",
            "months": "month", "micrograms": "microgram"}


def numeric_units(text: str) -> set[tuple[str, str]]:
    result = set()
    for value, unit in NUMBER_UNIT.findall(text):
        number = format(Decimal(value).normalize(), "f")
        normalized_unit = SINGULAR.get(unit.casefold(), unit.casefold())
        result.add((number, normalized_unit))
    return result


def passage_blocks(prompt: str) -> dict[str, str]:
    matches = list(PASSAGE_MARKER.finditer(prompt))
    return {match.group(1): prompt[match.end():matches[i + 1].start() if i + 1 < len(matches)
                                        else len(prompt)].strip()
            for i, match in enumerate(matches)}


def audit_one(prompt: dict, response: dict) -> dict:
    passages = passage_blocks(prompt["user_prompt"])
    parsed = response.get("parsed_response") or {}
    flags = []
    claims = parsed.get("claims") if isinstance(parsed.get("claims"), list) else []
    for index, claim in enumerate(claims):
        if not isinstance(claim, dict):
            continue
        text = claim.get("text", "")
        citations = claim.get("citations", [])
        if not isinstance(text, str) or not isinstance(citations, list):
            continue
        missing_ids = [citation for citation in citations if citation not in passages]
        if missing_ids:
            flags.append({"kind": "citation_not_in_visible_prompt", "claim_index": index,
                          "citations": missing_ids})
        cited_text = " ".join(passages[citation] for citation in citations if citation in passages)
        unsupported_numbers = sorted(numeric_units(text) - numeric_units(cited_text))
        if unsupported_numbers:
            flags.append({"kind": "claim_numeric_unit_absent_from_citations", "claim_index": index,
                          "numbers": unsupported_numbers})
    if parsed.get("abstain") is False and isinstance(parsed.get("answer"), str):
        claim_numbers = numeric_units(" ".join(claim.get("text", "") for claim in claims
                                               if isinstance(claim, dict) and isinstance(claim.get("text"), str)))
        unmatched = sorted(numeric_units(parsed["answer"]) - claim_numbers)
        if unmatched:
            flags.append({"kind": "answer_numeric_unit_absent_from_claims", "numbers": unmatched})
    return {"qid": prompt["qid"], "flags": flags, "claim_count": len(claims),
            "visible_passages": len(passages)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = {row["qid"]: row for row in read_jsonl(args.prompts)}
    responses = {row["qid"]: row for row in read_jsonl(args.responses)}
    if set(prompts) != set(responses):
        raise ValueError("Prompt and response QIDs differ")
    validate_provenance(prompts, responses)
    items = [audit_one(prompt, responses[qid]) for qid, prompt in prompts.items()]
    counts = Counter(flag["kind"] for item in items for flag in item["flags"])
    report = {
        "status": "mechanical_triage_only_not_clinical_claim_verification",
        "prompts_sha256": sha256(args.prompts.read_bytes()).hexdigest(),
        "responses_sha256": sha256(args.responses.read_bytes()).hexdigest(),
        "items": items, "flag_counts": dict(counts),
        "limitations": [
            "Only literal digit-plus-unit pairs are checked; synonyms, spelled-out numbers, negation, conditions and semantic entailment are not validated.",
            "A matching number or valid citation ID does not prove a claim is supported.",
            "Flags may be false positives and require source review before any accuracy calculation.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "items": len(items), "flag_counts": dict(counts)}))


if __name__ == "__main__":
    main()
