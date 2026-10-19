"""Assemble the public data archive (for example a Zenodo deposit) with SHA-256 sums.

The archive holds the frozen DailyMed snapshots and pinned XML, holdout questions, both
AI annotations and the AI audit, the AI silver standard, locks, prompts, responses and
reports. It does not copy the FDARxBench benchmark itself, which is cited at its pinned
commit; benchmark predictions are included keyed by question identifier. Private review
drafts, reviewer bundles and placeholder files are never copied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from .export_reviewer_interface import ROOT


GROUPS = {
    "holdout/sources": ["data/snapshots/version_holdout_sources.jsonl",
                        "data/snapshots/version_change_cohort_sources.jsonl",
                        "data/snapshots/version_expansion_consecutive_sources.jsonl",
                        "data/snapshots/version_expansion_change_sources.jsonl"],
    "holdout/pinned_xml": ["results/pinned_xml_holdout_after_gold/*.xml"],
    "holdout/design": ["data/holdout/version_test_manifest.json", "data/holdout/version_change_cohort_manifest.json",
                       "data/holdout/version_expansion_manifest.json",
                       "data/holdout/version_question_candidates.jsonl",
                       "data/holdout/version_question_candidates_expanded.jsonl",
                       "data/holdout/version_question_candidates_expanded.manifest.json",
                       "results/holdout_source_preflight_expanded.json",
                       "results/holdout_design_sensitivity_expanded.json"],
    "holdout/annotations": ["private_review/llm_annotator_claude.jsonl",
                            "private_review/llm_annotator_claude.provenance.json",
                            "private_review/llm_annotation_spec.json",
                            "data/review/codex_ai_expert_simulation.jsonl",
                            "data/review/codex_ai_expert_simulation.provenance.json",
                            "private_review/codex_ai_audit_2026-10-19.md",
                            "private_review/codex_ai_audit_2026-10-19.flags.json"],
    "holdout/silver_standard": ["data/holdout/version_silver_ai.jsonl", "data/holdout/version_silver_ai.manifest.json"],
    "holdout/locked_run": ["paper/ANALYSIS_PLAN.md", "paper/HOLDOUT_EXECUTION_LOCK.json",
                           "results/holdout_r2_*_prompts.jsonl", "results/holdout_r2_*_prompts.manifest.json",
                           "results/holdout_r2_*_responses.jsonl", "results/holdout_r2_*_responses.sessions.jsonl",
                           "results/holdout_retrieval_summary.json", "results/holdout_mechanical_scores.json"],
    "holdout/superseded_run1": ["paper/HOLDOUT_EXECUTION_LOCK_run1_superseded.json",
                                "results/holdout_bm25_*.jsonl", "results/holdout_bm25_prompts.manifest.json",
                                "results/holdout_learned_cover_*.jsonl",
                                "results/holdout_learned_cover_prompts.manifest.json",
                                "results/holdout_ontology_cover_*.jsonl",
                                "results/holdout_ontology_cover_prompts.manifest.json"],
    "benchmark": ["paper/BENCHMARK_TEST_LOCK.json", "results/benchmark_test_retrieval.json",
                  "results/benchmark_test_predictions.jsonl", "results/learned_selector_model.json",
                  "results/learned_selector_validation.json", "results/learned_selector_dev_crossfit.json",
                  "results/dev_learned_crossfit_predictions.jsonl", "results/validation_learned_predictions.jsonl",
                  "results/dense_validation.json", "results/dense_validation_predictions.jsonl",
                  "results/validation_retrieval_lexical.json", "results/validation_retrieval_greedy_cover.json",
                  "results/retrieval_error_analysis.json"],
    "benchmark/ablations": ["results/clause_coverage_ablation.json", "results/deep_pool_ablation.json",
                            "results/set_aware_ablation.json", "results/joint_pair_ablation.json",
                            "results/method_development_audit.json", "results/candidate_pool_diagnostics.json",
                            "results/retrieval_gate_validation.json", "results/crossfit_retrieval_gate.json",
                            "results/headroom_dev.json", "results/headroom_validation.json"],
    "development_pilots": ["results/paired_dev_qa_four_method_comparison.json",
                           "results/paired_dev_qa_repeat_sensitivity.json",
                           "results/version_table_span_aligned_comparison.json",
                           "results/synthetic_claim_judge_pilot.json", "results/spl_table_structure_audit.json"],
    "software_rehearsal": ["results/holdout_pipeline_rehearsal_v6_ai_silver.json"],
}

README = """# Data archive: complete multi-section evidence retrieval from FDA drug labels

Companion data for the manuscript "Learning to retrieve complete multi-section evidence from
FDA drug labels: a preregistered benchmark test and a descriptive temporal holdout".

Code: https://github.com/MinhNhat-2504/version-aware-label-rag (commit {commit}, MIT License).
Built {built} UTC. Verify with: `sha256sum -c SHA256SUMS.txt`.

## Important: no human review

No human domain expert reviewed the temporal holdout. `holdout/silver_standard` is an **AI silver
standard**, not gold: a fixed rule combined two disclosed AI annotations (Claude and Codex) that
are not independent (Codex audited the Claude annotation before annotating). Holdout results are
descriptive and carry no clinical claim. Nothing here is medical advice.

## Contents

- `holdout/sources`: frozen DailyMed SPL version snapshots (section text, URLs, ZIP/XML hashes).
- `holdout/pinned_xml`: pinned SPL XML for the 116 retained versions.
- `holdout/design`: sampling manifests, candidate questions (AI-drafted), preflight and design sensitivity.
- `holdout/annotations`: both AI annotations with provenance receipts, and the AI audit with its flags.
- `holdout/silver_standard`: the AI silver standard and its manifest (rule, exclusions, agreement).
- `holdout/locked_run`: analysis plan, execution lock, prompts, model responses and reports of the reported run.
- `holdout/superseded_run1`: the first locked run, superseded before any score was computed; not reported.
- `benchmark`: preregistration lock, one-shot test report and predictions, selector weights,
  development cross-fit and validation reports with per-question predictions.
- `benchmark/ablations`, `development_pilots`, `software_rehearsal`: exploratory and software-check reports.

## Licences and sources

Original annotations, reports and code outputs: CC BY 4.0. FDARxBench (Apache License 2.0) is not
redistributed; obtain it at commit 019785d2802c4ceec583e4d4b82b902d7a8f9910 from
https://github.com/xiongbetty/FDARxBench. Benchmark prediction files contain only question
identifiers and ranked passage indices. SPL documents were obtained from DailyMed (U.S. National
Library of Medicine); see the DailyMed terms of use.
"""


def expand(pattern: str, root: Path) -> list[Path]:
    matches = sorted(root.glob(pattern)) if any(ch in pattern for ch in "*?[") else [root / pattern]
    missing = [str(path) for path in matches if not path.is_file()]
    if not matches or missing:
        raise FileNotFoundError(f"Archive input missing: {pattern} {missing}")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("results/data_archive"))
    parser.add_argument("--name", default="version-aware-label-rag-data")
    args = parser.parse_args()
    root = args.root.resolve()
    folder = (args.output_dir if args.output_dir.is_absolute() else root / args.output_dir) / args.name
    archive = folder.with_suffix(".zip")
    if folder.exists() or archive.exists():
        parser.error("Archive already exists; choose a new --name")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "unknown"
    sums = []
    for target, patterns in GROUPS.items():
        for pattern in patterns:
            for source in expand(pattern, root):
                destination = folder / target / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                sums.append(f"{sha256(destination.read_bytes()).hexdigest()}  {destination.relative_to(folder).as_posix()}")
    (folder / "README.md").write_text(README.format(commit=commit, built=datetime.now(timezone.utc)
                                                    .strftime("%Y-%m-%d %H:%M")), encoding="utf-8", newline="\n")
    (folder / "SHA256SUMS.txt").write_text("\n".join(sorted(sums, key=lambda line: line[66:])) + "\n",
                                           encoding="utf-8", newline="\n")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                bundle.write(path, f"{args.name}/{path.relative_to(folder).as_posix()}")
    print(json.dumps({"folder": str(folder), "zip": str(archive), "files": len(sums),
                      "zip_bytes": archive.stat().st_size, "commit": commit}))


if __name__ == "__main__":
    main()
