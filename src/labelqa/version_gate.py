"""Counterfactual pre-answer gate applied to saved development responses."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from .data import read_jsonl


COADMIN = re.compile(r"\bco(?:-|\s*)administration\s+with\s+([a-z][a-z0-9-]*)\b", re.I)


def target_entity(question: str) -> str | None:
    match = COADMIN.search(question)
    return match.group(1) if match else None


def absent_from_label(entity: str, sections: list[dict]) -> bool:
    expression = re.compile(rf"\b{re.escape(entity)}\b", re.I)
    return not any(expression.search(section["text"]) for section in sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/snapshots/version_pair_paxlovid.jsonl"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair = next(read_jsonl(args.source))
    versions = {row["spl_version"]: row for row in pair["versions"]}
    prompts = {row["qid"]: row for row in read_jsonl(args.prompts)}
    responses = {row["qid"]: row for row in read_jsonl(args.responses)}
    if set(prompts) != set(responses):
        raise ValueError("Prompt and response QIDs differ")
    output, counts = [], Counter()
    for qid, prompt in prompts.items():
        source = versions[prompt["spl_version"]]
        if (prompt["set_id"] != pair["set_id"] or
                prompt["xml_sha256"] != source["xml_sha256"]):
            raise ValueError(f"Pinned source mismatch for {qid}")
        original = responses[qid]
        entity = target_entity(prompt["question"])
        if entity and absent_from_label(entity, source["sections"]):
            answer = (f"The parsed sections of this pinned SPL version do not mention {entity}; "
                      "the interaction cannot be determined from these sections.")
            parsed = {"answer": answer, "claims": [], "abstain": True}
            revised = {key: value for key, value in original.items()
                       if key not in ("raw_response", "parsed_response", "error")}
            revised.update({"raw_response": json.dumps(parsed), "parsed_response": parsed,
                            "answer_source": "counterfactual_deterministic_version_gate",
                            "original_response_sha256": sha256(json.dumps(original, sort_keys=True).encode()).hexdigest(),
                            "gate_target_entity": entity})
            counts["gated_absent_entity"] += 1
            output.append(revised)
        else:
            revised = dict(original)
            revised["answer_source"] = "original_model_response"
            revised["gate_target_entity"] = entity
            counts["passed_to_model_output"] += 1
            output.append(revised)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": dict(counts)}))


if __name__ == "__main__":
    main()
