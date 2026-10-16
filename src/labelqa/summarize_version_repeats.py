"""Summarize fixed-seed repeated calls on the two PAXLOVID development facts."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .score_qa import score_one


ROOT = Path("results")
RUNS = {
    "bm25_windows": ("version_pilot_bm25_chunked_prompts.jsonl",
                     "version_pilot_bm25_chunked_seeded_responses.jsonl",
                     "version_pilot_bm25_chunked_seeded_repeat5.jsonl"),
    "bm25_rows": ("version_pilot_bm25_table_audited_prompts.jsonl",
                  "version_pilot_bm25_table_audited_seeded_responses.jsonl",
                  "version_pilot_bm25_table_audited_seeded_repeat5.jsonl"),
    "evidence_set_rows": ("version_pilot_evidence_set_table_audited_prompts.jsonl",
                          "version_pilot_evidence_set_table_audited_seeded_responses.jsonl",
                          "version_pilot_evidence_set_table_audited_seeded_repeat5.jsonl"),
}


def summarize_run(prompt_name: str, single_name: str, repeat_name: str) -> dict:
    prompt_path, single_path, repeat_path = (ROOT / n for n in (prompt_name, single_name, repeat_name))
    prompts = {p["qid"]: p for p in read_jsonl(prompt_path) if p["task"] == "factual"}
    single = {r["qid"]: r for r in read_jsonl(single_path) if r["qid"] in prompts}
    grouped = defaultdict(list)
    for row in read_jsonl(repeat_path):
        grouped[row["qid"]].append(row)
    if set(prompts) != set(single) or set(prompts) != set(grouped):
        raise ValueError("Prompt, single-run, and repeat QIDs differ")
    items = {}
    run_digests = set()
    run_options = set()
    for qid, prompt in prompts.items():
        expected_hash = sha256((prompt["system_prompt"] + prompt["user_prompt"]).encode()).hexdigest()
        rows = grouped[qid]
        all_rows = [single[qid], *rows]
        if any(r.get("prompt_sha256") != expected_hash or r.get("error") for r in all_rows):
            raise ValueError(f"Prompt mismatch or model error for {qid}")
        if len({r["model_digest"] for r in all_rows}) != 1 or len({json.dumps(r["decoding_options"], sort_keys=True) for r in all_rows}) != 1:
            raise ValueError(f"Model or decoding mismatch for {qid}")
        run_digests.update(r["model_digest"] for r in all_rows)
        run_options.update(json.dumps(r["decoding_options"], sort_keys=True) for r in all_rows)
        if sorted(r["repeat"] for r in rows) != list(range(1, 6)):
            raise ValueError(f"Missing or duplicate repeat for {qid}")
        metrics = [score_one(prompt, row) for row in rows]
        single_metrics = score_one(prompt, single[qid])
        items[qid] = {
            "single_call": {"abstain": single[qid]["parsed_response"].get("abstain"),
                            "valid_schema": single_metrics["valid_schema"],
                            "correct_decision": single_metrics.get("correct_refusal_decision")},
            "repeat_calls": len(rows),
            "abstain_counts": dict(Counter(str(r["parsed_response"].get("abstain")).lower() for r in rows)),
            "valid_schema_count": sum(m["valid_schema"] for m in metrics),
            "correct_decision_count": sum(m.get("correct_refusal_decision") is True for m in metrics),
            "distinct_parsed_response_count": len({json.dumps(r["parsed_response"], sort_keys=True)
                                                    for r in rows}),
            "prompt_sha256": expected_hash,
        }
    if len(run_digests) != 1 or len(run_options) != 1:
        raise ValueError("Questions within a run use different model settings")
    return {
        "model_digest": next(iter(run_digests)),
        "decoding_options": json.loads(next(iter(run_options))),
        "prompt_file_sha256": sha256(prompt_path.read_bytes()).hexdigest(),
        "single_response_file_sha256": sha256(single_path.read_bytes()).hexdigest(),
        "repeat_response_file_sha256": sha256(repeat_path.read_bytes()).hexdigest(),
        "items": items,
    }


def main() -> None:
    runs = {name: summarize_run(*paths) for name, paths in RUNS.items()}
    if (len({r["model_digest"] for r in runs.values()}) != 1
            or len({json.dumps(r["decoding_options"], sort_keys=True) for r in runs.values()}) != 1
            or next(iter(runs.values()))["decoding_options"].get("seed") != 20261015):
        raise ValueError("Repeated runs did not use the same pinned model and fixed seed")
    report = {
        "status": "five_fixed_seed_repeats_per_answerable_question_and_method_development_only",
        "seed": 20261015,
        "runs": runs,
        "limitation": "Repeated calls characterize only these exact development prompts and local runtime. No independent clinical answer assessment, population error rate, or sealed-holdout result is inferred.",
    }
    output = ROOT / "version_table_seeded_repeatability.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": {
        name: {qid: {key: value for key, value in item.items()
                     if key in ("single_call", "abstain_counts", "valid_schema_count",
                                "correct_decision_count", "distinct_parsed_response_count")}
               for qid, item in run["items"].items()}
        for name, run in runs.items()}}))


if __name__ == "__main__":
    main()
