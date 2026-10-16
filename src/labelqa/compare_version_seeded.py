"""Compare frozen PAXLOVID development runs with matched model settings."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .score_qa import score_one, validate_provenance


ROOT = Path("results")
RUNS = {
    "bm25_windows_seeded": ("version_pilot_bm25_chunked_prompts.jsonl",
                            "version_pilot_bm25_chunked_seeded_responses.jsonl"),
    "bm25_rows_seeded": ("version_pilot_bm25_table_audited_prompts.jsonl",
                         "version_pilot_bm25_table_audited_seeded_responses.jsonl"),
    "evidence_set_rows_seeded": ("version_pilot_evidence_set_table_audited_prompts.jsonl",
                                 "version_pilot_evidence_set_table_audited_seeded_responses.jsonl"),
    "bm25_rows_live_gate_seeded": ("version_pilot_bm25_table_audited_prompts.jsonl",
                                   "version_pilot_bm25_table_audited_live_gate_responses.jsonl"),
}


def load_run(prompt_name: str, response_name: str) -> tuple[dict, dict, dict]:
    prompt_path, response_path = ROOT / prompt_name, ROOT / response_name
    prompts = {r["qid"]: r for r in read_jsonl(prompt_path)}
    responses = {r["qid"]: r for r in read_jsonl(response_path)}
    if set(prompts) != set(responses):
        raise ValueError(f"Unpaired QIDs in {response_name}")
    validate_provenance(prompts, responses)
    metadata = {
        "prompt_file": str(prompt_path), "response_file": str(response_path),
        "prompt_file_sha256": sha256(prompt_path.read_bytes()).hexdigest(),
        "response_file_sha256": sha256(response_path.read_bytes()).hexdigest(),
    }
    return prompts, responses, metadata


def main() -> None:
    report_runs = {}
    loaded = {}
    qid_reference = None
    model_digests = set()
    decoding_options = set()
    for name, (prompt_name, response_name) in RUNS.items():
        prompts, responses, metadata = load_run(prompt_name, response_name)
        if qid_reference is None:
            qid_reference = set(prompts)
        if set(prompts) != qid_reference:
            raise ValueError(f"QID mismatch in {name}")
        if any(r.get("error") for r in responses.values()):
            raise ValueError(f"Model error in {name}")
        model_digests.update(r["model_digest"] for r in responses.values())
        decoding_options.update(json.dumps(r["decoding_options"], sort_keys=True)
                                for r in responses.values())
        details = {qid: score_one(prompts[qid], responses[qid]) for qid in prompts}
        report_runs[name] = {
            **metadata,
            "n": len(prompts),
            "mean_prompt_words": round(sum(len(p["user_prompt"].split()) for p in prompts.values())
                                       / len(prompts), 1),
            "model_calls": sum(r.get("model_called", True) for r in responses.values()),
            "correct_decisions": sum(d.get("correct_refusal_decision") is True for d in details.values()),
            "valid_schema": sum(d["valid_schema"] for d in details.values()),
            "valid_schema_and_correct_decision": sum(
                d["valid_schema"] and d.get("correct_refusal_decision") is True
                for d in details.values()),
            "answerable_complete_retrieval": sum(d.get("retrieval_complete_evidence") is True
                                                   for d in details.values() if d["task"] == "factual"),
            "max_retrieved_passages": max(len(p["retrieved_indices"]) for p in prompts.values()),
            "items": {qid: {
                "task": details[qid]["task"],
                "correct_decision": details[qid].get("correct_refusal_decision"),
                "valid_schema": details[qid]["valid_schema"],
                "retrieval_complete_evidence": details[qid].get("retrieval_complete_evidence"),
                "cited_complete_evidence": details[qid].get("cited_complete_evidence"),
                "answer_source": responses[qid].get("answer_source", "model_response"),
                "model_called": responses[qid].get("model_called", True),
            } for qid in prompts},
        }
        loaded[name] = (prompts, responses)
    if len(model_digests) != 1 or len(decoding_options) != 1:
        raise ValueError("Model digest or decoding options differ across matched runs")
    if any(run["max_retrieved_passages"] > 5 for run in report_runs.values()):
        raise ValueError("A run exceeded the five-passage budget")
    options = json.loads(next(iter(decoding_options)))
    if options.get("seed") != 20261015:
        raise ValueError("Unexpected seed for matched experiment")
    row_prompts, row_answers = loaded["bm25_rows_seeded"]
    gate_prompts, gate_answers = loaded["bm25_rows_live_gate_seeded"]
    if any((row_prompts[qid]["system_prompt"], row_prompts[qid]["user_prompt"])
           != (gate_prompts[qid]["system_prompt"], gate_prompts[qid]["user_prompt"])
           for qid in qid_reference):
        raise ValueError("Gate and no-gate prompts differ")
    changed = [qid for qid in sorted(qid_reference) if gate_answers[qid].get("model_called")
               and row_answers[qid].get("parsed_response") != gate_answers[qid].get("parsed_response")]
    report = {
        "status": "matched_settings_four_question_assistant_drafted_development_pilot",
        "model_digest": next(iter(model_digests)), "decoding_options": options,
        "same_top_k": 5,
        "runs": report_runs,
        "same_prompt_model_called_outputs_differed_between_gate_and_no_gate": changed,
        "limitations": [
            "One generation per question and method; matched seed does not ensure identical outputs on repeat calls.",
            "Prompt lengths and evidence representation differ; only model digest, decoding options and top-five passage cap were matched.",
            "Mechanical decisions and citation IDs do not establish complete, clinically correct, source-supported answers.",
            "The four questions and gate were developed after inspecting this label; no sealed holdout was opened.",
        ],
    }
    output = ROOT / "version_table_seeded_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": {
        name: {key: run[key] for key in ("correct_decisions", "valid_schema",
                                       "valid_schema_and_correct_decision", "model_calls")}
        for name, run in report_runs.items()}, "changed_same_prompt_outputs": changed}))


if __name__ == "__main__":
    main()
