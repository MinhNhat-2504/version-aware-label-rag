"""Record hashes of paper-facing results without copying large local artifacts.

The ``human_review`` field is derived from the holdout receipts rather than typed by
hand: it stays "pending" until compiled manual gold exists and an end-to-end answer
score was computed from that same gold and execution lock.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


REPORTS = (
    "results/validation_retrieval_lexical.json",
    "results/learned_selector_dev_crossfit.json",
    "results/learned_selector_model.json",
    "results/learned_selector_validation.json",
    "results/benchmark_test_retrieval.json",
    "results/benchmark_test_predictions.jsonl",
    "results/dense_validation.json",
    "results/retrieval_gate_validation.json",
    "results/crossfit_retrieval_gate.json",
    "results/clause_coverage_ablation.json",
    "results/candidate_pool_diagnostics.json",
    "results/deep_pool_ablation.json",
    "results/set_aware_ablation.json",
    "results/joint_pair_ablation.json",
    "results/method_development_audit.json",
    "results/holdout_design_sensitivity.json",
    "results/holdout_design_sensitivity_expanded.json",
    "results/retrieval_error_analysis.json",
    "results/paired_dev_qa_manifest.json",
    "results/paired_dev_qa_comparison.json",
    "results/paired_dev_qa_repeatability.json",
    "results/paired_dev_qa_dense_manifest.json",
    "results/paired_dev_qa_four_method_comparison.json",
    "results/paired_dev_qa_repeat_sensitivity.json",
    "results/paired_dev_qa_bm25_scores.json",
    "results/paired_dev_qa_learned_cover_scores.json",
    "results/paired_dev_qa_bm25_repeat_scores.json",
    "results/paired_dev_qa_learned_cover_repeat_scores.json",
    "results/paired_dev_qa_dense_scores.json",
    "results/paired_dev_qa_bm25_dense_rrf_scores.json",
    "results/paired_dev_qa_dense_repeat_scores.json",
    "results/paired_dev_qa_bm25_dense_rrf_repeat_scores.json",
    "results/paired_dev_qa_bm25_numeric_audit.json",
    "results/paired_dev_qa_learned_cover_numeric_audit.json",
    "results/paired_dev_qa_dense_numeric_audit.json",
    "results/paired_dev_qa_bm25_dense_rrf_numeric_audit.json",
    "results/version_table_seeded_comparison.json",
    "results/version_table_seeded_repeatability.json",
    "results/spl_table_structure_audit.json",
    "results/version_table_span_aligned_comparison.json",
    "results/synthetic_claim_judge_pilot.json",
    "results/holdout_source_preflight.json",
    "results/holdout_source_preflight_expanded.json",
    "results/holdout_pipeline_rehearsal_v3.json",
    "results/equivalence_adjudicated_pilot_coverage.json",
)
SOURCES = (
    "data/upstream/FDARxBench/data/labels/labels.jsonl",
    "data/upstream/FDARxBench/data/qa/qa.jsonl",
    "paper/BENCHMARK_TEST_LOCK.json",
    "data/holdout/version_question_candidates.jsonl",
    "data/holdout/version_test_manifest.json",
    "data/holdout/version_change_cohort_manifest.json",
    "data/holdout/version_expansion_manifest.json",
    "data/holdout/version_question_candidates_expanded.jsonl",
    "data/holdout/version_question_candidates_expanded.manifest.json",
    "results/equivalence_review_validation.jsonl",
    "data/review/equivalence_validation_manifest.json",
    "data/review/equivalence_validation_reader_a_blank.jsonl",
    "data/review/equivalence_validation_reader_b_blank.jsonl",
)


HOLDOUT_GOLD = "data/holdout/version_gold_adjudicated.jsonl"
HOLDOUT_GOLD_MANIFEST = "data/holdout/version_gold_adjudicated.manifest.json"
HOLDOUT_LOCK = "paper/HOLDOUT_EXECUTION_LOCK.json"
HOLDOUT_RETRIEVAL = "results/holdout_retrieval_summary.json"
HOLDOUT_SCORES = "results/holdout_answer_scores.json"
HOLDOUT_ARTIFACTS = (HOLDOUT_GOLD, HOLDOUT_GOLD_MANIFEST, HOLDOUT_LOCK, HOLDOUT_RETRIEVAL, HOLDOUT_SCORES)
PENDING = "pending; no adjudicated gold or external holdout model scores"


def read_json(path: str) -> dict | None:
    file = Path(path)
    return json.loads(file.read_text(encoding="utf-8")) if file.is_file() else None


def signers(design: dict | None) -> str:
    design = design or {}
    llm = design.get("llm_annotator_ids") or []
    text = f"{design.get('resolution_mode')} signed by {', '.join(design.get('resolver_ids') or [])}"
    return text + (f"; disclosed LLM annotator {', '.join(llm)}" if llm else "")


def human_review_status() -> str:
    """Derive the review state from receipts; 'pending' until scores match gold and lock."""
    gold = read_json(HOLDOUT_GOLD_MANIFEST)
    if gold is None or not Path(HOLDOUT_GOLD).is_file():
        return PENDING
    from .review_resolution import review_design
    gold_text = f"gold compiled ({signers(review_design(gold))})"
    scores, lock = read_json(HOLDOUT_SCORES), Path(HOLDOUT_LOCK)
    if scores is None or not lock.is_file():
        return f"{gold_text}; holdout execution or answer review pending"
    if (scores.get("status") != "manual_adjudicated_end_to_end_answer_score_after_gold"
            or scores.get("gold_sha256") != gold.get("gold_sha256")
            or scores.get("execution_lock_sha256") != sha256(lock.read_bytes()).hexdigest()):
        return f"{gold_text}; answer scores do not match gold and lock, review pending"
    return f"complete: {gold_text}; answer reviews {signers(scores.get('answer_review_design'))}"


def benchmark_lock_code() -> dict:
    """Show whether code hashed by the one-shot benchmark lock has changed since the test."""
    lock = read_json("paper/BENCHMARK_TEST_LOCK.json") or {}
    changed = sorted(name for name, digest in (lock.get("code_sha256") or {}).items()
                     if sha256((Path("src/labelqa") / name).read_bytes()).hexdigest() != digest)
    return {"lock_created_utc": lock.get("created_utc"), "files_changed_since_lock": changed,
            "note": ("The test ran once under the locked hashes; later edits do not change its saved "
                     "report but mean HEAD no longer reproduces the lock check.") if changed else
                    "Locked code matches the working tree."}


def describe(path: str) -> dict:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(file)
    return {"sha256": sha256(file.read_bytes()).hexdigest(), "bytes": file.stat().st_size}


def main() -> None:
    report = {
        "status": "working_paper_local_artifact_hashes_not_final_release",
        "artifacts": {path: describe(path) for path in REPORTS},
        "sources": {path: describe(path) for path in SOURCES},
        "holdout_artifacts": {path: describe(path) for path in HOLDOUT_ARTIFACTS if Path(path).is_file()},
        "human_review": human_review_status(),
        "benchmark_test": "retrieval proxy evaluated once after paper/BENCHMARK_TEST_LOCK.json; no answer generation",
        "benchmark_test_lock_code": benchmark_lock_code(),
        "note": "The results directory and upstream snapshot are local/ignored; retain files with these hashes for reproduction.",
    }
    output = Path("paper/RESULTS_MANIFEST.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "artifacts": len(REPORTS), "sources": len(SOURCES)}))


if __name__ == "__main__":
    main()
