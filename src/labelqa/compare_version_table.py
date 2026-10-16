"""Summarize mechanical PAXLOVID window-versus-row pilot results."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


ROOT = Path("results")
RUNS = {
    "bm25_160_word_windows": ("version_pilot_bm25_chunked_prompts.jsonl",
                              "version_pilot_bm25_chunked_responses.jsonl",
                              "version_pilot_bm25_chunked_scores.json"),
    "bm25_atomic_table_rows_audited": ("version_pilot_bm25_table_audited_prompts.jsonl",
                                       "version_pilot_bm25_table_audited_responses.jsonl",
                                       "version_pilot_bm25_table_audited_scores.json"),
    "evidence_set_atomic_table_rows_audited": ("version_pilot_evidence_set_table_audited_prompts.jsonl",
                                               "version_pilot_evidence_set_table_audited_responses.jsonl",
                                               "version_pilot_evidence_set_table_audited_scores.json"),
    "bm25_rows_counterfactual_absence_gate_audited": ("version_pilot_bm25_table_audited_prompts.jsonl",
                                                        "version_pilot_bm25_table_audited_counterfactual_responses.jsonl",
                                                        "version_pilot_bm25_table_audited_counterfactual_scores.json"),
    "bm25_rows_live_gate_seeded_audited": ("version_pilot_bm25_table_audited_prompts.jsonl",
                                            "version_pilot_bm25_table_audited_live_gate_responses.jsonl",
                                            "version_pilot_bm25_table_audited_live_gate_scores.json"),
}


def main() -> None:
    rows = {}
    qid_reference = None
    model_digests = set()
    for name, (prompt_name, response_name, score_name) in RUNS.items():
        prompt_path, response_path, score_path = (ROOT / prompt_name, ROOT / response_name, ROOT / score_name)
        prompts = list(read_jsonl(prompt_path))
        responses = list(read_jsonl(response_path))
        scores = json.loads(score_path.read_text(encoding="utf-8"))["details"]
        qids = {row["qid"] for row in prompts}
        if qid_reference is None:
            qid_reference = qids
        if qids != qid_reference or qids != {r["qid"] for r in responses} or qids != {d["qid"] for d in scores}:
            raise ValueError(f"Unpaired pilot QIDs in {name}")
        model_digests.update(r["model_digest"] for r in responses)
        answerable = [d for d in scores if d["task"] == "factual"]
        rows[name] = {
            "n": len(prompts),
            "mean_user_prompt_words": round(sum(len(p["user_prompt"].split()) for p in prompts) / len(prompts), 1),
            "answerable_complete_retrieval": sum(d.get("retrieval_complete_evidence") is True for d in answerable),
            "correct_refusal_decisions": sum(d.get("correct_refusal_decision") is True for d in scores),
            "valid_schema": sum(d.get("valid_schema") is True for d in scores),
            "model_calls": sum(r.get("model_called", True) for r in responses),
            "prompt_sha256": sha256(prompt_path.read_bytes()).hexdigest(),
            "response_sha256": sha256(response_path.read_bytes()).hexdigest(),
            "score_sha256": sha256(score_path.read_bytes()).hexdigest(),
        }
    if len(model_digests) != 1:
        raise ValueError("Comparison used different answering model digests")
    report = {
        "status": "four_question_assistant_drafted_development_pilot_not_independent_evaluation",
        "same_model_digest": next(iter(model_digests)),
        "source_pair": "data/snapshots/version_pair_paxlovid.jsonl",
        "methods": rows,
        "caveat": "Audited row_v2 parser skips empty source rows. All four row prompt hashes changed from the pre-audit parser. The 160-word baseline and unseeded row runs share decoding options but not prompt representation; the live gate uses a fixed seed, so its generation results are not a controlled passage-only contrast. Mechanical evidence/citation metrics do not establish answer completeness or clinical correctness. The counterfactual gate substitutes saved responses; the live gate calls the model only for v12. Both were designed after inspecting this development pilot. Prior row results and the provisional error ledger describe superseded prompts.",
    }
    output = ROOT / "version_table_comparison_audited.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "methods": {
        name: {key: value for key, value in result.items() if not key.endswith("sha256")}
        for name, result in rows.items()}}))


if __name__ == "__main__":
    main()
