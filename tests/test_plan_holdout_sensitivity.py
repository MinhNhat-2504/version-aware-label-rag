import unittest

from labelqa.plan_holdout_sensitivity import (
    build_report,
    exact_calibrated_items,
    exact_mcnemar_p,
    optimistic_exact_power,
)


class HoldoutSensitivityPlanTests(unittest.TestCase):
    def setUp(self):
        self.preflight = {
            "pairs": 12,
            "versions": 24,
            "candidate_questions": 24,
            "independent_human_adjudications": 0,
            "holdout_model_evaluations": 0,
            "cohorts": {
                "data\\snapshots\\version_holdout_sources.jsonl": 8,
                "data\\snapshots\\version_change_cohort_sources.jsonl": 4,
            },
        }
        self.expanded = {
            **self.preflight, "pairs": 60, "versions": 120, "candidate_questions": 120,
            "cohorts": {
                "data/snapshots/version_holdout_sources.jsonl": 8,
                "data/snapshots/version_change_cohort_sources.jsonl": 4,
                "data/snapshots/version_expansion_consecutive_sources.jsonl": 32,
                "data/snapshots/version_expansion_change_sources.jsonl": 16,
            },
        }
        self.error_analysis = {"questions": 559, "counts": {
            "learned_cover_win_over_bm25": 31, "learned_cover_loss_to_bm25": 18}}

    def test_exact_mcnemar_all_win_boundary(self):
        self.assertAlmostEqual(exact_mcnemar_p(5, 0), 0.0625)
        self.assertAlmostEqual(exact_mcnemar_p(6, 0), 0.03125)

    def test_power_increases_with_sample_size_and_effect(self):
        small = optimistic_exact_power(24, 0.30, 0.70)
        larger_sample = optimistic_exact_power(96, 0.30, 0.70)
        larger_effect = optimistic_exact_power(24, 0.30, 0.90)
        self.assertGreater(larger_sample, small)
        self.assertGreater(larger_effect, small)

    def test_exact_calibration_reaches_target(self):
        n, power = exact_calibrated_items(0.30, 0.75, 0.80)
        self.assertGreaterEqual(power, 0.80)
        self.assertGreater(n, 24)

    def test_report_uses_metadata_only_and_labels_scope(self):
        report = build_report(self.preflight, b"analysis plan")
        self.assertEqual(report["holdout_results_read"], 0)
        self.assertEqual(
            report["interpretation"]["current_claim_scope"],
            "feasibility_and_descriptive_paired_evaluation",
        )
        self.assertEqual(
            report["exact_paired_binary_reference"][
                "minimum_all_candidate_wins_with_zero_losses_for_p_below_alpha"],
            6,
        )

    def test_expanded_design_pools_waves_into_sampling_strata(self):
        report = build_report(self.expanded, b"analysis plan", self.error_analysis)
        resolution = report["design_resolution"]
        self.assertEqual(resolution["label_pairs_by_stratum"],
                         {"consecutive_pairs": 40, "text_change_enriched": 20})
        self.assertAlmostEqual(resolution["one_item_percentage_points"], 0.8333)
        self.assertEqual(resolution["one_item_in_text_change_enriched_stratum_percentage_points"], 2.5)

    def test_empirical_effect_reference_is_far_from_feasible(self):
        report = build_report(self.expanded, b"analysis plan", self.error_analysis)
        reference = report["empirical_validation_effect_reference"]
        self.assertAlmostEqual(reference["implied_absolute_difference"], 0.0233, places=4)
        self.assertGreater(reference["approximate_independent_items_for_80_percent_power"], 1000)
        self.assertLess(reference["optimistic_power_at_current_items"], 0.2)

    def test_report_rejects_unpaired_design(self):
        self.expanded["versions"] = 119
        with self.assertRaises(ValueError):
            build_report(self.expanded, b"analysis plan")

    def test_report_refuses_post_outcome_generation(self):
        self.preflight["holdout_model_evaluations"] = 1
        with self.assertRaises(ValueError):
            build_report(self.preflight, b"analysis plan")


if __name__ == "__main__":
    unittest.main()
