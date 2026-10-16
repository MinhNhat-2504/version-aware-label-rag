"""Record hashes of paper-facing results without copying large local artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


REPORTS = (
    "results/validation_retrieval_lexical.json",
    "results/learned_selector_dev_crossfit.json",
    "results/learned_selector_model.json",
    "results/learned_selector_validation.json",
    "results/dense_validation.json",
    "results/retrieval_gate_validation.json",
    "results/clause_coverage_ablation.json",
    "results/candidate_pool_diagnostics.json",
    "results/deep_pool_ablation.json",
    "results/set_aware_ablation.json",
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
    "results/equivalence_adjudicated_pilot_coverage.json",
)
SOURCES = (
    "data/upstream/FDARxBench/data/labels/labels.jsonl",
    "data/upstream/FDARxBench/data/qa/qa.jsonl",
    "data/holdout/version_question_candidates.jsonl",
    "data/holdout/version_test_manifest.json",
    "data/holdout/version_change_cohort_manifest.json",
    "results/equivalence_review_validation.jsonl",
    "data/review/equivalence_validation_manifest.json",
    "data/review/equivalence_validation_reader_a_blank.jsonl",
    "data/review/equivalence_validation_reader_b_blank.jsonl",
)


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
        "human_review": "pending; no adjudicated gold or external holdout model scores",
        "benchmark_test": "not evaluated",
        "note": "The results directory and upstream snapshot are local/ignored; retain files with these hashes for reproduction.",
    }
    output = Path("paper/RESULTS_MANIFEST.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "artifacts": len(REPORTS), "sources": len(SOURCES)}))


if __name__ == "__main__":
    main()
