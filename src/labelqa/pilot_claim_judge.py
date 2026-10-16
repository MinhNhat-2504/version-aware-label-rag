"""Synthetic-only feasibility probe for a cited-claim relation judge."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .run_local import model_digest, post_json


MODEL = "qwen2.5:3b"
SYSTEM = ("Judge one factual claim against the single cited source passage only. "
          "Return JSON with verdict (supported, contradicted, or insufficient), "
          "quote (an exact short substring of the source, or empty), and reason (short string). "
          "Supported requires the same entity, relation, quantity, condition and negation. "
          "Contradicted requires an explicit incompatible source statement. "
          "If the source is silent, ambiguous, or about another entity, say insufficient. "
          "Do not use medical knowledge outside this passage.")
CASES = (
    ("suzetrigine_supported", "Coadministration with suzetrigine is contraindicated.",
     "Coadministration with suzetrigine is contraindicated.", "supported"),
    ("wrong_drug_insufficient", "Coadministration with suzetrigine is contraindicated.",
     "Coadministration with mavacamten is contraindicated.", "insufficient"),
    ("mavacamten_contra_insufficient",
     "Mavacamten: Discontinue during PAXLOVID treatment. Do not restart earlier than 5 days after completion.",
     "Mavacamten is contraindicated with PAXLOVID.", "insufficient"),
    ("restart_timing_contradicted",
     "Mavacamten: Discontinue during PAXLOVID treatment. Do not restart earlier than 5 days after completion.",
     "Mavacamten may be restarted immediately after treatment.", "contradicted"),
    ("dose_contradicted", "Take 2 mg once daily.", "Take 4 mg once daily.", "contradicted"),
    ("negation_contradicted", "No dose adjustment is recommended in mild renal impairment.",
     "Dose adjustment is recommended in mild renal impairment.", "contradicted"),
    ("population_insufficient", "The trial excluded pregnant patients.",
     "This medication is safe in pregnancy.", "insufficient"),
    ("relation_supported", "Drug A increases exposure to Drug B; monitor for adverse reactions.",
     "Drug A increases exposure to Drug B.", "supported"),
    ("relation_contradicted", "Drug A increases exposure to Drug B; monitor for adverse reactions.",
     "Drug A decreases exposure to Drug B.", "contradicted"),
    ("condition_insufficient", "Avoid use with severe hepatic impairment.",
     "Avoid use with mild hepatic impairment.", "insufficient"),
)


def parse_verdict(raw: str, source: str) -> dict:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Invalid relation-judge schema")
    verdict = parsed.get("verdict")
    quote = parsed.get("quote")
    reason = parsed.get("reason")
    if (verdict not in ("supported", "contradicted", "insufficient")
            or not isinstance(quote, str) or not isinstance(reason, str)):
        raise ValueError("Invalid relation-judge schema")
    if verdict != "insufficient" and not quote:
        raise ValueError("Decisive verdict requires an exact source quote")
    if quote and quote not in source:
        raise ValueError("Judge quote does not occur in cited source")
    return {"verdict": verdict, "quote": quote, "reason": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_claim_judge_pilot.json"))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.output.exists():
        parser.error("Positive repeats and a new output path are required")
    digest = model_digest(MODEL)
    options = {"temperature": 0, "seed": 20261016, "num_ctx": 8192, "num_predict": 256}
    items = []
    for case_id, source, claim, expected in CASES:
        user = f"Source passage:\n{source}\n\nClaim:\n{claim}"
        calls = []
        for repeat in range(args.repeats):
            response = post_json("/api/chat", {"model": MODEL,
                                                "messages": [{"role": "system", "content": SYSTEM},
                                                             {"role": "user", "content": user}],
                                                "stream": False, "format": "json", "options": options}, 180)
            raw = response.get("message", {}).get("content", "")
            try:
                result = parse_verdict(raw, source)
                calls.append({"parsed": result, "matches_synthetic_expected": result["verdict"] == expected,
                              "raw": raw})
            except (ValueError, json.JSONDecodeError) as exc:
                calls.append({"error": str(exc), "raw": raw, "matches_synthetic_expected": False})
        items.append({"case_id": case_id, "source": source, "claim": claim,
                      "synthetic_expected": expected, "prompt_sha256": sha256((SYSTEM + user).encode()).hexdigest(),
                      "calls": calls})
        print(json.dumps({"case": case_id, "done": len(items)}), flush=True)
    report = {
        "status": "synthetic_claim_relation_feasibility_only_not_clinical_validation",
        "model": MODEL, "model_digest": digest, "decoding_options": options,
        "system_prompt_sha256": sha256(SYSTEM.encode()).hexdigest(),
        "cases": len(CASES), "repeats": args.repeats,
        "matching_synthetic_cases_per_repeat": [
            sum(item["calls"][i]["matches_synthetic_expected"] for item in items)
            for i in range(args.repeats)],
        "verdict_changed_between_first_two_calls": sum(
            item["calls"][0].get("parsed", {}).get("verdict") != item["calls"][1].get("parsed", {}).get("verdict")
            for item in items) if args.repeats >= 2 else None,
        "items": items,
        "limitations": ["Cases are constructed from short synthetic snippets; they are not independent expert labels.",
                        "A quote existing in the source does not prove the verdict is semantically correct.",
                        "Do not deploy this judge or use it to score the sealed holdout without human calibration."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "matching": report["matching_synthetic_cases_per_repeat"],
                      "changed": report["verdict_changed_between_first_two_calls"]}))


if __name__ == "__main__":
    main()
