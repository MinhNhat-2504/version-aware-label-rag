"""Prospective sensitivity analysis from holdout metadata, without model outcomes."""

from __future__ import annotations

import argparse
from hashlib import sha256
from functools import lru_cache
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .holdout_reporting import cohort_for


PREFLIGHT = Path("results/holdout_source_preflight.json")
ANALYSIS_PLAN = Path("paper/ANALYSIS_PLAN.md")
ERROR_ANALYSIS = Path("results/retrieval_error_analysis.json")
OUTPUT = Path("results/holdout_design_sensitivity.json")
ALPHA = 0.05


def binomial_probability(n: int, successes: int, probability: float) -> float:
    """Binomial mass in log space, so large planning sizes do not overflow."""
    if probability in (0.0, 1.0):
        return float(successes == (n if probability == 1.0 else 0))
    log_mass = (math.lgamma(n + 1) - math.lgamma(successes + 1) - math.lgamma(n - successes + 1)
                + successes * math.log(probability) + (n - successes) * math.log1p(-probability))
    return math.exp(log_mass)


@lru_cache(maxsize=None)
def exact_mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    """Two-sided exact McNemar p-value conditional on discordant items."""
    if candidate_only < 0 or baseline_only < 0:
        raise ValueError("Discordant counts must be nonnegative")
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    smaller = min(candidate_only, baseline_only)
    lower_tail = sum(binomial_probability(discordant, value, 0.5)
                     for value in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail)


@lru_cache(maxsize=None)
def critical_candidate_wins(discordant: int, alpha: float = ALPHA) -> int | None:
    """Smallest favorable discordant count giving a directional exact rejection."""
    for wins in range(discordant // 2 + 1, discordant + 1):
        if exact_mcnemar_p(wins, discordant - wins) <= alpha:
            return wins
    return None


def optimistic_exact_power(n: int, discordance_rate: float,
                           candidate_share: float, alpha: float = ALPHA) -> float:
    """Exact power assuming independent items; clustering can only weaken this guide."""
    if n < 1:
        raise ValueError("n must be positive")
    if not (0.0 <= discordance_rate <= 1.0 and 0.5 <= candidate_share <= 1.0):
        raise ValueError("Invalid planning probabilities")
    power = 0.0
    for discordant in range(n + 1):
        critical = critical_candidate_wins(discordant, alpha)
        if critical is None:
            continue
        discordant_probability = binomial_probability(n, discordant, discordance_rate)
        conditional_rejection = sum(
            binomial_probability(discordant, wins, candidate_share)
            for wins in range(critical, discordant + 1)
        )
        power += discordant_probability * conditional_rejection
    return power


def approximate_required_items(discordance_rate: float, candidate_share: float,
                               target_power: float, alpha: float = ALPHA) -> int:
    """Normal-approximation planning size for a paired binary difference."""
    difference = discordance_rate * (2 * candidate_share - 1)
    if difference <= 0 or not (0.5 < target_power < 1.0):
        raise ValueError("Planning requires a positive effect and power between 0.5 and 1")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(target_power)
    alternative_variance = discordance_rate - difference ** 2
    size = ((z_alpha * math.sqrt(discordance_rate)
             + z_power * math.sqrt(alternative_variance)) / difference) ** 2
    return math.ceil(size)


def exact_calibrated_items(discordance_rate: float, candidate_share: float,
                           target_power: float) -> tuple[int, float]:
    """Calibrate upward from the normal approximation to the exact power target."""
    n = approximate_required_items(discordance_rate, candidate_share, target_power)
    while optimistic_exact_power(n, discordance_rate, candidate_share) < target_power:
        n += 1
    return n, optimistic_exact_power(n, discordance_rate, candidate_share)


def power_grid(n: int) -> list[dict[str, Any]]:
    rows = []
    for discordance_rate in (0.15, 0.30, 0.50):
        for candidate_share in (0.60, 0.70, 0.80, 0.90):
            rows.append({
                "discordance_rate": discordance_rate,
                "candidate_share_among_discordances": candidate_share,
                "implied_absolute_difference": round(
                    discordance_rate * (2 * candidate_share - 1), 4),
                "optimistic_power": round(
                    optimistic_exact_power(n, discordance_rate, candidate_share), 4),
            })
    return rows


def stratum_pairs(preflight: dict[str, Any]) -> dict[str, int]:
    """Count frozen label pairs per prespecified sampling stratum."""
    counts: dict[str, int] = {}
    for source, pairs in preflight.get("cohorts", {}).items():
        stratum = cohort_for(source)
        counts[stratum] = counts.get(stratum, 0) + pairs
    if sum(counts.values()) != preflight.get("pairs"):
        raise ValueError("Cohort pair counts do not sum to the frozen design")
    return counts


def empirical_reference(error_analysis: dict[str, Any], items: int) -> dict[str, Any]:
    """Plan from the observed validation discordance of learned_cover versus BM25."""
    counts = error_analysis["counts"]
    wins = counts["learned_cover_win_over_bm25"]
    losses = counts["learned_cover_loss_to_bm25"]
    n = error_analysis["questions"]
    discordance = (wins + losses) / n
    share = wins / (wins + losses)
    return {
        "role": ("Observed effect on the inspected validation retrieval proxy; descriptive planning "
                 "input only, with the same optimistic item-independence assumption."),
        "validation_questions": n,
        "learned_only_successes": wins,
        "bm25_only_successes": losses,
        "discordance_rate": round(discordance, 4),
        "candidate_share_among_discordances": round(share, 4),
        "implied_absolute_difference": round(discordance * (2 * share - 1), 4),
        "approximate_independent_items_for_80_percent_power":
            approximate_required_items(discordance, share, 0.80),
        "approximate_independent_items_for_90_percent_power":
            approximate_required_items(discordance, share, 0.90),
        "optimistic_power_at_current_items": round(
            optimistic_exact_power(items, discordance, share), 4),
    }


def build_report(preflight: dict[str, Any], analysis_plan_bytes: bytes,
                 error_analysis: dict[str, Any] | None = None) -> dict[str, Any]:
    pairs = preflight.get("pairs")
    versions = preflight.get("versions")
    questions = preflight.get("candidate_questions")
    if (not isinstance(pairs, int) or pairs < 1 or versions != 2 * pairs
            or not isinstance(questions, int) or questions < versions):
        raise ValueError("Frozen holdout needs two versions per pair and one question per version")
    strata = stratum_pairs(preflight)
    if preflight.get("independent_human_adjudications") != 0:
        raise ValueError("Sensitivity plan must be generated before human adjudication")
    if preflight.get("holdout_model_evaluations") != 0:
        raise ValueError("Sensitivity plan must be generated before model evaluation")

    all_win_threshold = next(
        discordant for discordant in range(1, questions + 1)
        if exact_mcnemar_p(discordant, 0) <= ALPHA
    )
    planning_scenarios = []
    for absolute_difference in (0.10, 0.15, 0.20):
        discordance_rate = 0.30
        candidate_share = (absolute_difference / discordance_rate + 1.0) / 2.0
        n80 = approximate_required_items(discordance_rate, candidate_share, 0.80)
        n90 = approximate_required_items(discordance_rate, candidate_share, 0.90)
        calibrated80, calibrated_power80 = exact_calibrated_items(
            discordance_rate, candidate_share, 0.80)
        calibrated90, calibrated_power90 = exact_calibrated_items(
            discordance_rate, candidate_share, 0.90)
        planning_scenarios.append({
            "assumed_discordance_rate": discordance_rate,
            "candidate_share_among_discordances": round(candidate_share, 6),
            "implied_absolute_difference": absolute_difference,
            "approximate_independent_items_for_80_percent_power": n80,
            "exact_power_at_approximate_80_percent_n": round(
                optimistic_exact_power(n80, discordance_rate, candidate_share), 4),
            "exact_calibrated_independent_items_for_80_percent_power": calibrated80,
            "exact_power_at_calibrated_80_percent_n": round(calibrated_power80, 4),
            "approximate_independent_items_for_90_percent_power": n90,
            "exact_power_at_approximate_90_percent_n": round(
                optimistic_exact_power(n90, discordance_rate, candidate_share), 4),
            "exact_calibrated_independent_items_for_90_percent_power": calibrated90,
            "exact_power_at_calibrated_90_percent_n": round(calibrated_power90, 4),
        })

    return {
        "status": "prospective_design_sensitivity_metadata_only_no_holdout_outcomes",
        "generated_before": {
            "independent_human_adjudication": True,
            "holdout_model_evaluation": True,
        },
        "inputs": {
            "label_pairs": pairs,
            "version_items": versions,
            "candidate_questions": questions,
            "primary_independent_bootstrap_clusters_at_most": pairs,
            "preflight_sha256": sha256(
                json.dumps(preflight, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "analysis_plan_sha256": sha256(analysis_plan_bytes).hexdigest(),
        },
        "design_resolution": {
            "one_item_percentage_points": round(100 / questions, 4),
            **{f"one_item_in_{stratum}_stratum_percentage_points": round(100 / (2 * count), 4)
               for stratum, count in sorted(strata.items())},
            "label_pairs_by_stratum": strata,
            "independent_label_pair_clusters": pairs,
        },
        "exact_paired_binary_reference": {
            "role": (
                "Optimistic item-independent sensitivity reference only; the declared analysis "
                "uses label-pair clustered bootstrap and may have less precision."
            ),
            "alpha_two_sided": ALPHA,
            "minimum_all_candidate_wins_with_zero_losses_for_p_below_alpha": all_win_threshold,
            "p_value_five_wins_zero_losses": round(exact_mcnemar_p(5, 0), 6),
            "p_value_six_wins_zero_losses": round(exact_mcnemar_p(6, 0), 6),
            "current_item_power_grid": power_grid(questions),
            "label_pair_cluster_sign_abstraction_power_grid": power_grid(pairs),
        },
        "optimistic_item_independent_planning_scenarios": planning_scenarios,
        "empirical_validation_effect_reference": (
            empirical_reference(error_analysis, questions) if error_analysis else None),
        "interpretation": {
            "current_claim_scope": "feasibility_and_descriptive_paired_evaluation",
            "superiority_claim": (
                "The current design cannot be described as adequately powered for modest "
                "superiority effects. A non-significant result is inconclusive."
            ),
            "expansion_rule": (
                "If superiority is retained as a goal, freeze a separately selected larger "
                "label-pair cohort before any human gold or model output is inspected."
            ),
            "attrition": (
                "Reviewer exclusions reduce the accepted denominator and precision; the report "
                "must be regenerated from metadata before execution if the frozen design expands."
            ),
        },
        "limitations": [
            "The exact-power grid assumes independent items and is optimistic for two versions clustered within a label pair.",
            "Discordance rates and conditional win shares are planning assumptions, not observed holdout effects.",
            "The enriched four-pair cohort is not a random sample and does not support population prevalence claims.",
            "No human gold, retrieval prediction, answer or holdout endpoint is read by this analysis.",
        ],
        "holdout_results_read": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--error-analysis", type=Path, default=ERROR_ANALYSIS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a frozen design-sensitivity report")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    error_analysis = json.loads(args.error_analysis.read_text(encoding="utf-8"))
    report = build_report(preflight, ANALYSIS_PLAN.read_bytes(), error_analysis)
    report["inputs"]["preflight_path"] = args.preflight.as_posix()
    report["inputs"]["error_analysis_sha256"] = sha256(args.error_analysis.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "status": report["status"],
        "pairs": report["inputs"]["label_pairs"],
        "items": report["inputs"]["candidate_questions"],
        "all_win_threshold": report["exact_paired_binary_reference"][
            "minimum_all_candidate_wins_with_zero_losses_for_p_below_alpha"],
        "claim_scope": report["interpretation"]["current_claim_scope"],
        "holdout_results_read": report["holdout_results_read"],
    }))


if __name__ == "__main__":
    main()
