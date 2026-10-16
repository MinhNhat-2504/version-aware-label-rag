"""Compare the span-aligned PAXLOVID development rerun with preserved inputs."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .score_qa import score_one, validate_provenance


ROOT = Path("results")
RUNS = {
    "bm25_rows": ("version_pilot_bm25_table_span_aligned_prompts.jsonl",
                  "version_pilot_bm25_table_span_aligned_seeded_responses.jsonl"),
    "evidence_set_rows": ("version_pilot_evidence_set_table_span_aligned_prompts.jsonl",
                         "version_pilot_evidence_set_table_span_aligned_seeded_responses.jsonl"),
    "bm25_rows_live_gate": ("version_pilot_bm25_table_span_aligned_prompts.jsonl",
                            "version_pilot_bm25_table_span_aligned_live_gate_seeded_responses.jsonl"),
}
OLD_PROMPTS = {
    "bm25": ROOT / "version_pilot_bm25_table_audited_prompts.jsonl",
    "evidence_set": ROOT / "version_pilot_evidence_set_table_audited_prompts.jsonl",
}
OUTPUT = ROOT / "version_table_span_aligned_comparison.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def indexed(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    if not rows or len({row["qid"] for row in rows}) != len(rows):
        raise ValueError(f"Missing or duplicate pilot items: {path}")
    return {row["qid"]: row for row in rows}


def summarize() -> dict:
    runs = {}
    prompt_sets = {}
    identities = set()
    for name, (prompt_name, response_name) in RUNS.items():
        prompt_path, response_path = ROOT / prompt_name, ROOT / response_name
        prompts, responses = indexed(prompt_path), indexed(response_path)
        if set(prompts) != set(responses):
            raise ValueError(f"Prompt/response IDs differ: {name}")
        validate_provenance(prompts, responses)
        if any(row.get("error") for row in responses.values()):
            raise ValueError(f"Pilot generation error: {name}")
        if any(prompt.get("parser_version") != "row_v3_span_aligned" for prompt in prompts.values()):
            raise ValueError(f"Wrong parser version: {name}")
        for response in responses.values():
            identities.add((response["model_digest"], json.dumps(response["decoding_options"], sort_keys=True)))
        scores = {qid: score_one(prompts[qid], responses[qid]) for qid in prompts}
        runs[name] = {
            "prompt_file": str(prompt_path), "prompt_sha256": digest(prompt_path),
            "response_file": str(response_path), "response_sha256": digest(response_path),
            "items": len(scores),
            "answerable_complete_retrieval": sum(
                score["retrieval_complete_evidence"] is True for score in scores.values()
                if score["task"] == "factual"),
            "correct_refusal_decisions": sum(
                score["correct_refusal_decision"] is True for score in scores.values()),
            "valid_schema": sum(score["valid_schema"] for score in scores.values()),
            "joint_schema_and_decision": sum(
                score["valid_schema"] and score["correct_refusal_decision"] is True
                for score in scores.values()),
            "model_calls": sum(row.get("model_called", True) for row in responses.values()),
            "per_item": {qid: {"task": score["task"],
                               "valid_schema": score["valid_schema"],
                               "correct_refusal_decision": score["correct_refusal_decision"],
                               "retrieval_complete_evidence": score.get("retrieval_complete_evidence")}
                         for qid, score in scores.items()},
        }
        prompt_sets[name] = prompts
    if len(identities) != 1:
        raise ValueError("Reruns have different model digests or decoding options")
    model_hash, option_json = next(iter(identities))
    options = json.loads(option_json)
    if options.get("seed") != 20261015:
        raise ValueError("Unexpected development seed")
    if any(set(prompts) != set(prompt_sets["bm25_rows"]) for prompts in prompt_sets.values()):
        raise ValueError("Methods have different pilot questions")
    if prompt_sets["bm25_rows"] != prompt_sets["bm25_rows_live_gate"]:
        raise ValueError("Gate comparison did not use identical prompts")
    changes = {}
    for method, run in (("bm25", "bm25_rows"), ("evidence_set", "evidence_set_rows")):
        old, new = indexed(OLD_PROMPTS[method]), prompt_sets[run]
        if set(old) != set(new):
            raise ValueError(f"Old/new question sets differ: {method}")
        changes[method] = {
            "old_prompt_file": str(OLD_PROMPTS[method]),
            "old_prompt_sha256": digest(OLD_PROMPTS[method]),
            "changed_user_prompts": sorted(qid for qid in old if old[qid]["user_prompt"] != new[qid]["user_prompt"]),
            "changed_rankings": sorted(qid for qid in old if old[qid]["retrieved_indices"] != new[qid]["retrieved_indices"]),
        }
    return {
        "status": "span_aligned_parser_development_rerun_not_external_validation",
        "model_digest": model_hash, "decoding_options": options,
        "source": "Pinned PAXLOVID v11/v12 development XML only",
        "prompt_changes": changes, "runs": runs,
        "limitations": ["Mechanical schema, decision and citation-ID metrics do not validate claim meaning or clinical correctness.",
                        "Four assistant-drafted development items cannot establish method superiority.",
                        "Old row-model outputs remain archived under their original prompt hashes and must not be pooled with this rerun.",
                        "No sealed version holdout source or model output was used."],
    }


def main() -> None:
    if OUTPUT.exists():
        raise ValueError(f"Refusing to overwrite comparison: {OUTPUT}")
    report = summarize()
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": {
        name: {key: run[key] for key in ("answerable_complete_retrieval",
                                             "correct_refusal_decisions", "valid_schema",
                                             "joint_schema_and_decision", "model_calls")}
        for name, run in report["runs"].items()}}))


if __name__ == "__main__":
    main()
