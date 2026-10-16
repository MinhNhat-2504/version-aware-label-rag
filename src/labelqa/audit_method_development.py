"""Build a fail-closed ledger for post-canonical retrieval ablations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


OUTPUT = Path("results/method_development_audit.json")
EXPECTED_DEV_N = 2393
EXPECTED_VALIDATION_N = 559
RATE_TOLERANCE = 0.00011


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    path: Path
    baseline_key: str
    candidate_key: str
    comparison_path: tuple[str, ...]
    requires_nonzero_final_weight: bool = False


EXPERIMENTS = (
    ExperimentSpec(
        "clause_coverage",
        Path("results/clause_coverage_ablation.json"),
        "learned_cover",
        "clause_cover_ablation",
        ("dev_crossfit", "comparison"),
        True,
    ),
    ExperimentSpec(
        "deep_pool",
        Path("results/deep_pool_ablation.json"),
        "canonical_learned_cover",
        "deep_pool_learned_cover",
        ("dev_crossfit", "comparison"),
    ),
    ExperimentSpec(
        "set_aware",
        Path("results/set_aware_ablation.json"),
        "canonical_learned_cover",
        "deep_set_aware",
        ("dev_crossfit", "comparisons", "set_aware_minus_canonical"),
    ),
    ExperimentSpec(
        "joint_pair",
        Path("results/joint_pair_ablation.json"),
        "canonical_learned_cover",
        "joint_pair",
        ("dev_crossfit", "joint_minus_canonical", "bootstrap"),
    ),
)


def nested(mapping: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing report field: {'.'.join(path)}")
        value = value[key]
    return value


def complete_count(metric: dict[str, Any]) -> int:
    n = metric.get("n")
    rate = metric.get("all_gold@5")
    if not isinstance(n, int) or not isinstance(rate, (int, float)):
        raise ValueError("Metric must contain integer n and numeric all_gold@5")
    return round(n * float(rate))


def audit_experiment(spec: ExperimentSpec, report: dict[str, Any]) -> dict[str, Any]:
    """Validate one ablation and apply the dev-first future governance rule."""
    status = report.get("status", "")
    if "validation_descriptive" not in status or "holdout" not in status:
        raise ValueError(f"{spec.name}: report status does not mark descriptive validation/holdout")
    dev_metrics = nested(report, ("dev_crossfit", "metrics"))
    validation_metrics = nested(report, ("validation", "metrics"))
    try:
        dev_baseline = dev_metrics[spec.baseline_key]
        dev_candidate = dev_metrics[spec.candidate_key]
        validation_baseline = validation_metrics[spec.baseline_key]
        validation_candidate = validation_metrics[spec.candidate_key]
    except KeyError as error:
        raise ValueError(f"{spec.name}: missing method metric {error.args[0]}") from error
    if dev_baseline["n"] != EXPECTED_DEV_N or dev_candidate["n"] != EXPECTED_DEV_N:
        raise ValueError(f"{spec.name}: development denominator changed")
    if (validation_baseline["n"] != EXPECTED_VALIDATION_N
            or validation_candidate["n"] != EXPECTED_VALIDATION_N):
        raise ValueError(f"{spec.name}: validation denominator changed")

    comparison = nested(report, spec.comparison_path)
    delta = float(dev_candidate["all_gold@5"]) - float(dev_baseline["all_gold@5"])
    reported_delta = float(comparison["candidate_minus_baseline"])
    if abs(delta - reported_delta) > RATE_TOLERANCE:
        raise ValueError(f"{spec.name}: development comparison is inconsistent with metrics")
    interval = comparison.get("ci95_label_bootstrap")
    if (not isinstance(interval, list) or len(interval) != 2
            or not all(isinstance(value, (int, float)) for value in interval)):
        raise ValueError(f"{spec.name}: invalid development confidence interval")
    if comparison.get("n_questions") != EXPECTED_DEV_N:
        raise ValueError(f"{spec.name}: comparison denominator changed")

    reasons = []
    if reported_delta <= 0:
        reasons.append("development_point_estimate_not_positive")
    if float(interval[0]) <= 0:
        reasons.append("development_interval_not_strictly_positive")
    selected_weight = None
    if spec.requires_nonzero_final_weight:
        selected_weight = nested(report, ("full_dev_selection", "selected_weight"))
        if float(selected_weight) == 0.0:
            reasons.append("full_development_fit_reverts_to_canonical")

    decision = "retain_canonical" if reasons else "eligible_for_future_confirmation"
    return {
        "experiment": spec.name,
        "source_report": spec.path.as_posix(),
        "source_report_canonical_json_sha256": sha256(
            json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "development": {
            "n": EXPECTED_DEV_N,
            "baseline_complete": complete_count(dev_baseline),
            "candidate_complete": complete_count(dev_candidate),
            "candidate_minus_baseline": reported_delta,
            "ci95_label_bootstrap": interval,
        },
        "validation_descriptive_only": {
            "n": EXPECTED_VALIDATION_N,
            "baseline_complete": complete_count(validation_baseline),
            "candidate_complete": complete_count(validation_candidate),
        },
        "selected_full_development_weight": selected_weight,
        "decision": decision,
        "decision_reasons": reasons,
    }


def build_audit(specs: tuple[ExperimentSpec, ...] = EXPERIMENTS) -> dict[str, Any]:
    reports = {}
    entries = []
    for spec in specs:
        if not spec.path.is_file():
            raise FileNotFoundError(spec.path)
        report = json.loads(spec.path.read_text(encoding="utf-8"))
        reports[spec.name] = report
        entry = audit_experiment(spec, report)
        entry["source_file_sha256"] = sha256(spec.path.read_bytes()).hexdigest()
        entries.append(entry)

    label_hashes = {report["provenance"]["labels_sha256"] for report in reports.values()}
    qa_hashes = {report["provenance"]["qa_sha256"] for report in reports.values()}
    if len(label_hashes) != 1 or len(qa_hashes) != 1:
        raise ValueError("Ablation reports do not use one common benchmark snapshot")
    dev_baselines = {entry["development"]["baseline_complete"] for entry in entries}
    validation_baselines = {
        entry["validation_descriptive_only"]["baseline_complete"] for entry in entries
    }
    if dev_baselines != {1525} or validation_baselines != {343}:
        raise ValueError("Canonical selector counts changed across ablations")

    return {
        "status": "pass_retrospective_method_development_ledger_no_holdout",
        "scope": "post_canonical_retrieval_ablations",
        "policy_timing": (
            "This ledger was created after the listed exploratory experiments. It is a "
            "transparency and future-governance control, not a preregistration or multiplicity correction."
        ),
        "future_promotion_rule": {
            "primary_split": "label-grouped development cross-fit",
            "required_conditions": [
                "candidate development all-gold@5 point estimate is positive versus canonical",
                "label-bootstrap 95% interval has a strictly positive lower bound",
                "a tuned full-development fit must not revert to the canonical setting",
            ],
            "validation_role": "descriptive_only_and_never_sufficient_for_promotion",
            "automatic_promotion": False,
        },
        "canonical_counts": {
            "development_crossfit_complete": 1525,
            "development_n": EXPECTED_DEV_N,
            "validation_complete": 343,
            "validation_n": EXPECTED_VALIDATION_N,
        },
        "benchmark_snapshot": {
            "labels_sha256": next(iter(label_hashes)),
            "qa_sha256": next(iter(qa_hashes)),
        },
        "experiments": entries,
        "summary": {
            "experiments_audited": len(entries),
            "retain_canonical": sum(
                entry["decision"] == "retain_canonical" for entry in entries),
            "eligible_for_future_confirmation": sum(
                entry["decision"] == "eligible_for_future_confirmation" for entry in entries),
            "holdout_results_read": 0,
        },
    }


def main() -> None:
    audit = build_audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "status": audit["status"],
        **audit["summary"],
    }))


if __name__ == "__main__":
    main()
