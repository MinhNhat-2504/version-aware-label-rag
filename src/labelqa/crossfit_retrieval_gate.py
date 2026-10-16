"""Select a retrieval gate threshold from out-of-fold dev scores with a clustered risk bound."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

from .calibrate_retrieval_gate import (
    DEV_PREDICTIONS,
    FEATURE_NAMES,
    SOURCE,
    TARGET_DEV_RISK,
    VALIDATION_PREDICTIONS,
    calibration_diagnostics,
    cluster_bootstrap_selective,
    features,
    fit,
    probability,
    selective,
)
from .crossfit_learned import FOLDS, fold_of
from .data import load_labels, load_questions, read_jsonl, split_label
from .retrieval import LabelIndex


OUTPUT = Path("results/crossfit_retrieval_gate.json")
MODEL_OUTPUT = Path("results/crossfit_retrieval_gate_model.json")
PRIOR_GATE_REPORT = Path("results/retrieval_gate_validation.json")
ALPHA = 0.05
MIN_ACCEPTED = 50
COVERAGE_GRID = tuple(step / 100 for step in range(1, 101))
BOUND_DRAWS = 2000
SEED = 20261017


def crossfit_scores(examples: list[tuple[str, str, int, tuple[float, ...], bool]]) -> list[float]:
    """Score each dev question with a gate fitted only on the other label folds."""
    folds = sorted({fold for _, _, fold, _, _ in examples})
    if len(folds) < 2:
        raise ValueError("Cross-fitting needs at least two label folds")
    scores: list[float | None] = [None] * len(examples)
    for fold in folds:
        training = [(values, correct) for _, _, other, values, correct in examples if other != fold]
        weights = fit(training)
        for position, (_, _, other, values, _) in enumerate(examples):
            if other == fold:
                scores[position] = probability(weights, values)
    if any(score is None for score in scores):
        raise ValueError("A development question received no out-of-fold score")
    return [float(score) for score in scores]


def threshold_grid(scores: list[float], coverages: tuple[float, ...] = COVERAGE_GRID) -> list[float]:
    """Thresholds at fixed coverage quantiles, ordered from most to least conservative."""
    if not scores:
        raise ValueError("Threshold grid needs scores")
    ordered = sorted(scores, reverse=True)
    thresholds = []
    for coverage in coverages:
        if not 0.0 < coverage <= 1.0:
            raise ValueError("Coverage grid values must lie in (0, 1]")
        threshold = ordered[max(1, round(coverage * len(ordered))) - 1]
        if threshold not in thresholds:
            thresholds.append(threshold)
    return sorted(thresholds, reverse=True)


def clustered_upper_risk_bounds(scores: list[tuple[float, bool, str]], thresholds: list[float],
                                alpha: float = ALPHA, draws: int = BOUND_DRAWS,
                                seed: int = SEED) -> list[float | None]:
    """One-sided label-bootstrap upper bounds for missing-gold risk at every threshold.

    All thresholds share the same resampled labels, so the bounds are comparable
    within the prespecified fixed sequence.
    """
    if not 0.0 < alpha < 1.0 or draws < 1:
        raise ValueError("Bound needs alpha in (0, 1) and positive draws")
    by_label: dict[str, list[int]] = defaultdict(lambda: [0] * (2 * len(thresholds)))
    for score, correct, set_id in scores:
        counts = by_label[set_id]
        for position, threshold in enumerate(thresholds):
            if score >= threshold:
                counts[2 * position] += 1
                counts[2 * position + 1] += int(not correct)
    labels = sorted(by_label)
    if not labels:
        return [None] * len(thresholds)
    generator = random.Random(seed)
    samples: list[list[float]] = [[] for _ in thresholds]
    for _ in range(draws):
        totals = [0] * (2 * len(thresholds))
        for label in generator.choices(labels, k=len(labels)):
            totals = [left + right for left, right in zip(totals, by_label[label])]
        for position in range(len(thresholds)):
            accepted = totals[2 * position]
            # An empty resample cannot show acceptable risk; treat it as a failure.
            samples[position].append(totals[2 * position + 1] / accepted if accepted else 1.0)
    bounds = []
    for values in samples:
        values.sort()
        bounds.append(values[min(int((1.0 - alpha) * len(values)), len(values) - 1)])
    return bounds


def fixed_sequence_threshold(scores: list[tuple[float, bool, str]], target: float,
                             alpha: float = ALPHA, min_accepted: int = MIN_ACCEPTED,
                             draws: int = BOUND_DRAWS, seed: int = SEED) -> tuple[float, list[dict]]:
    """Widen coverage until the clustered upper risk bound first exceeds the target."""
    candidates = [threshold for threshold in threshold_grid([row[0] for row in scores])
                  if selective(scores, threshold)["accepted"] >= min_accepted]
    bounds = clustered_upper_risk_bounds(scores, candidates, alpha, draws, seed)
    selected = 1.01  # Abstain on every question if the first test fails.
    trace = []
    for threshold, bound in zip(candidates, bounds):
        result = selective(scores, threshold)
        passed = bound is not None and bound <= target
        trace.append({"threshold": threshold, "accepted": result["accepted"],
                      "coverage": result["coverage"],
                      "observed_missing_gold_risk": result["observed_missing_gold_risk"],
                      "upper_risk_bound": None if bound is None else round(bound, 4),
                      "passes": passed})
        if not passed:
            break
        selected = threshold
    return selected, trace


def build_examples(rows: list[dict], split: str, questions: dict, labels: dict,
                   indexes: dict[str, LabelIndex]) -> list[tuple[str, str, int, tuple[float, ...], bool]]:
    examples = []
    for row in rows:
        q = questions[row["qid"]]
        if split_label(q.set_id) != split or q.task != "multihop" or not (0 < len(q.gold) <= 5):
            raise ValueError(f"Prediction entered wrong split: {q.qid}")
        if tuple(row["gold_indices"]) != q.gold or row["set_id"] != q.set_id:
            raise ValueError(f"Upstream gold or label changed for {q.qid}")
        fold = row.get("fold", -1)
        if split == "dev" and fold != fold_of(q.set_id):
            raise ValueError(f"Development fold differs from label grouping: {q.qid}")
        if q.set_id not in indexes:
            indexes[q.set_id] = LabelIndex(labels[q.set_id])
        ranking = row["rankings"]["learned_cover"]
        if len(ranking) > 5 or len(set(ranking)) != len(ranking):
            raise ValueError(f"Invalid retrieval budget for {q.qid}")
        examples.append((q.qid, q.set_id, fold, features(indexes[q.set_id], q.text, ranking),
                         set(q.gold) <= set(ranking)))
    if len({qid for qid, *_ in examples}) != len(examples):
        raise ValueError(f"Duplicate prediction in {split}")
    return examples


def main() -> None:
    labels = load_labels(SOURCE / "labels/labels.jsonl")
    questions = {q.qid: q for q in load_questions(SOURCE / "qa/qa.jsonl")}
    indexes: dict[str, LabelIndex] = {}
    dev = build_examples(list(read_jsonl(DEV_PREDICTIONS)), "dev", questions, labels, indexes)
    if {fold for _, _, fold, _, _ in dev} != set(range(FOLDS)):
        raise ValueError("Development predictions do not cover every label fold")

    oof = crossfit_scores(dev)
    dev_oof_scores = [(score, correct, set_id)
                      for score, (_, set_id, _, _, correct) in zip(oof, dev)]
    threshold, trace = fixed_sequence_threshold(dev_oof_scores, TARGET_DEV_RISK)
    weights = fit([(values, correct) for _, _, _, values, correct in dev])
    model = {"fit_split": "dev grouped crossfit outputs only",
             "threshold_source": "out-of-fold dev scores with fixed-sequence clustered upper risk bound",
             "features": list(FEATURE_NAMES), "weights": [round(value, 8) for value in weights],
             "threshold": threshold, "target_missing_gold_risk": TARGET_DEV_RISK,
             "alpha_one_sided": ALPHA, "min_accepted": MIN_ACCEPTED,
             "coverage_grid": [COVERAGE_GRID[0], COVERAGE_GRID[-1], len(COVERAGE_GRID)],
             "bound_draws": BOUND_DRAWS, "seed": SEED,
             "dev_predictions_sha256": sha256(DEV_PREDICTIONS.read_bytes()).hexdigest()}
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Validation is read only after the model and threshold files are written.
    validation = build_examples(list(read_jsonl(VALIDATION_PREDICTIONS)), "validation",
                                questions, labels, indexes)
    validation_scores = [(probability(weights, values), correct, set_id)
                         for _, set_id, _, values, correct in validation]
    dev_refit_scores = [(probability(weights, values), correct, set_id)
                        for _, set_id, _, values, correct in dev]
    validation_result = cluster_bootstrap_selective(validation_scores, threshold)
    validation_bound = clustered_upper_risk_bounds(validation_scores, [threshold])[0] \
        if validation_result["accepted"] else None
    prior = json.loads(PRIOR_GATE_REPORT.read_text(encoding="utf-8"))
    accepted_on_dev = selective(dev_oof_scores, threshold)["accepted"] > 0
    report = {
        "status": "crossfit_gate_threshold_dev_only_validation_descriptive_no_holdout",
        "model_sha256": sha256(MODEL_OUTPUT.read_bytes()).hexdigest(),
        "threshold_selected_on_dev_out_of_fold_scores": threshold,
        "selection_rule": {
            "target_missing_gold_risk": TARGET_DEV_RISK,
            "bound": "one-sided 95% label-cluster bootstrap percentile of accepted missing-gold risk",
            "sequence": "coverage quantiles from 1% to 100%, most conservative first; stop at first failure",
            "min_accepted": MIN_ACCEPTED,
        },
        "dev_out_of_fold": {**cluster_bootstrap_selective(dev_oof_scores, threshold),
                            "calibration": calibration_diagnostics(dev_oof_scores)},
        "dev_full_refit_in_sample": cluster_bootstrap_selective(dev_refit_scores, threshold),
        "fixed_sequence_trace": trace,
        "dev_out_of_fold_risk_coverage_curve_diagnostic_only": [
            {"threshold": round(value, 6), **selective(dev_oof_scores, value)}
            for value in threshold_grid([row[0] for row in dev_oof_scores])
        ],
        "validation_descriptive": {
            **validation_result,
            "one_sided_95_upper_risk_bound": None if validation_bound is None
            else round(validation_bound, 4),
            "point_risk_meets_target": (validation_result["observed_missing_gold_risk"] is not None
                                        and validation_result["observed_missing_gold_risk"]
                                        <= TARGET_DEV_RISK),
            "calibration": calibration_diagnostics(validation_scores),
        },
        "prior_point_estimate_gate": {
            "report": PRIOR_GATE_REPORT.as_posix(),
            "report_sha256": sha256(PRIOR_GATE_REPORT.read_bytes()).hexdigest(),
            "threshold": prior["threshold_selected_on_dev_only"],
            "validation_accepted": prior["validation"]["accepted"],
            "validation_observed_missing_gold_risk": prior["validation"]["observed_missing_gold_risk"],
        },
        "decision": ("eligible_for_future_confirmation_not_promoted" if accepted_on_dev
                     else "no_dev_threshold_meets_bound_gate_abstains_on_all"),
        "provenance": {
            "dev_predictions_sha256": sha256(DEV_PREDICTIONS.read_bytes()).hexdigest(),
            "validation_predictions_sha256": sha256(VALIDATION_PREDICTIONS.read_bytes()).hexdigest(),
            "gate_feature_code_sha256": sha256(
                Path(__file__).with_name("calibrate_retrieval_gate.py").read_bytes()).hexdigest(),
            "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Outcome is retrieval of all upstream gold indices, not claim support or clinical answer correctness.",
            "The redesign follows an observed validation failure of the earlier gate, so validation is adaptive and descriptive.",
            "The deployed weights are a full-dev refit; the threshold comes from out-of-fold scores of fold-specific fits.",
            "The bootstrap bound is approximate and not a finite-sample safety guarantee.",
            "Confirmation requires untouched data; the benchmark test split and temporal holdout remain unopened.",
        ],
        "holdout_results_read": 0,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "threshold": threshold,
                      "dev_out_of_fold": {key: report["dev_out_of_fold"][key] for key in
                                          ("accepted", "coverage", "observed_missing_gold_risk")},
                      "validation": {key: report["validation_descriptive"][key] for key in
                                     ("accepted", "coverage", "observed_missing_gold_risk",
                                      "one_sided_95_upper_risk_bound")},
                      "decision": report["decision"]}))


if __name__ == "__main__":
    main()
