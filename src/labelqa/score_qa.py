"""Mechanical QA/citation diagnostics; clinical correctness needs adjudication."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re


PASSAGE_ID = re.compile(r"^\[?P(\d{4})\]?$", re.I)
WORD = re.compile(r"\b\w+\b", re.UNICODE)


def token_f1(reference: str, predicted: str) -> float:
    from collections import Counter

    gold = Counter(WORD.findall(reference.lower()))
    guess = Counter(WORD.findall(predicted.lower()))
    overlap = sum((gold & guess).values())
    if not gold or not guess or not overlap:
        return 0.0
    precision = overlap / sum(guess.values())
    recall = overlap / sum(gold.values())
    return 2 * precision * recall / (precision + recall)


def score_one(prompt: dict, response: dict) -> dict:
    parsed = response.get("parsed_response")
    result = {"qid": prompt["qid"], "task": prompt["task"], "model_error": bool(response.get("error"))}
    if prompt.get("gold_claims"):
        retrieved = set(prompt["retrieved_indices"])
        result["retrieval_complete_evidence"] = all(
            retrieved & set(group) for group in prompt["gold_claims"]
        )
    if not isinstance(parsed, dict):
        result["valid_schema"] = False
        return result
    answer = parsed.get("answer")
    abstain = parsed.get("abstain")
    if isinstance(abstain, bool):
        result["abstain"] = abstain
        result["correct_refusal_decision"] = abstain == (prompt["task"] == "refusal")
    version = prompt.get("schema_version", 1)
    claims = parsed.get("claims") if version == 2 else None
    if version == 2:
        valid_claims = (
            isinstance(claims, list) and
            all(isinstance(claim, dict) and isinstance(claim.get("text"), str)
                and bool(claim["text"].strip()) and isinstance(claim.get("citations"), list)
                and all(isinstance(citation, str) for citation in claim["citations"])
                for claim in claims)
        )
        valid = (isinstance(answer, str) and bool(answer.strip()) and isinstance(abstain, bool) and valid_claims
                 and (not claims if abstain else bool(claims)))
        citations = [citation for claim in claims for citation in claim["citations"]] if valid_claims else []
    else:
        citations = parsed.get("citations")
        valid = (isinstance(answer, str) and isinstance(abstain, bool)
                 and isinstance(citations, list)
                 and all(isinstance(citation, str) for citation in citations))
    result["valid_schema"] = valid
    if not valid:
        return result
    indices = []
    malformed = 0
    for citation in citations:
        match = PASSAGE_ID.fullmatch(citation.strip())
        if match:
            indices.append(int(match.group(1)))
        else:
            malformed += 1
    retrieved = set(prompt["retrieved_indices"])
    gold = set(prompt["gold_indices"])
    result.update({
        "citation_count": len(citations),
        "invalid_citation_count": malformed + len(set(indices) - retrieved),
        "gold_citation_recall": (len(set(indices) & gold) / len(gold)
                                 if gold and not prompt.get("gold_claims") else None),
        "lexical_answer_f1": token_f1(prompt.get("reference_answer", ""), answer)
        if prompt["task"] != "refusal" else None,
        "claim_support_verified": False,
    })
    if prompt.get("gold_claims"):
        groups = [set(group) for group in prompt["gold_claims"]]
        result["cited_complete_evidence"] = all(set(indices) & group for group in groups)
    if version == 2:
        result["claim_count"] = len(claims)
        result["uncited_claim_count"] = sum(not claim["citations"] for claim in claims)
        result["claim_citation_rate"] = (sum(bool(claim["citations"]) for claim in claims)
                                         / len(claims)) if claims else None
    return result


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def validate_provenance(prompts: dict[str, dict], responses: dict[str, dict]) -> None:
    """Refuse to score an answer against a different frozen prompt."""
    for qid, prompt in prompts.items():
        response = responses[qid]
        expected = sha256((prompt["system_prompt"] + prompt["user_prompt"]).encode()).hexdigest()
        if response.get("prompt_sha256") != expected or response.get("task") != prompt["task"]:
            raise ValueError(f"Prompt hash or task mismatch for {qid}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = {item["qid"]: item for item in (json.loads(s) for s in args.prompts.read_text(encoding="utf-8").splitlines())}
    responses = {item["qid"]: item for item in (json.loads(s) for s in args.responses.read_text(encoding="utf-8").splitlines())}
    if set(prompts) != set(responses):
        raise ValueError(f"Prompt/response QID mismatch: missing={len(set(prompts)-set(responses))}, extra={len(set(responses)-set(prompts))}")
    validate_provenance(prompts, responses)
    details = [score_one(prompts[qid], responses[qid]) for qid in prompts]
    groups = defaultdict(list)
    for row in details:
        groups[row["task"]].append(row)
    metrics = {}
    for task, rows in sorted(groups.items()):
        valid = [row for row in rows if row["valid_schema"]]
        metrics[task] = {
            "n": len(rows),
            "schema_valid_rate": mean([float(row["valid_schema"]) for row in rows]),
            "refusal_decision_accuracy": mean([
                float(row["correct_refusal_decision"]) for row in rows
                if "correct_refusal_decision" in row
            ]),
            "mean_invalid_citations": mean([row["invalid_citation_count"] for row in valid]),
            "mean_gold_citation_recall": mean([
                row["gold_citation_recall"] for row in valid
                if row["gold_citation_recall"] is not None
            ]),
            "mean_lexical_answer_f1": mean([
                row["lexical_answer_f1"] for row in valid
                if row["lexical_answer_f1"] is not None
            ]),
            "mean_claim_citation_rate": mean([
                row["claim_citation_rate"] for row in valid
                if row.get("claim_citation_rate") is not None
            ]),
            "retrieval_complete_evidence_rate": mean([
                float(row["retrieval_complete_evidence"]) for row in rows
                if "retrieval_complete_evidence" in row
            ]),
            "cited_complete_evidence_rate": mean([
                float(row["cited_complete_evidence"]) for row in valid
                if "cited_complete_evidence" in row
            ]),
        }
    report = {
        "prompts": str(args.prompts),
        "responses": str(args.responses),
        "metrics": metrics,
        "details": details,
        "interpretation": "Mechanical diagnostics only; no claim-level or clinical correctness has been verified.",
        "provenance_checked": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
