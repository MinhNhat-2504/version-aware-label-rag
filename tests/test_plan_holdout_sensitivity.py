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
        }

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

    def test_report_refuses_post_outcome_generation(self):
        self.preflight["holdout_model_evaluations"] = 1
        with self.assertRaises(ValueError):
            build_report(self.preflight, b"analysis plan")


if __name__ == "__main__":
    unittest.main()
