import unittest

from labelqa.calibrate_retrieval_gate import (
    calibration_diagnostics,
    choose_threshold,
    cluster_bootstrap_selective,
    selective,
)


class RetrievalGateTests(unittest.TestCase):
    def test_threshold_is_selected_only_from_development_scores(self):
        scores = [(0.9, True)] * 20 + [(0.1, False)] * 20
        threshold = choose_threshold(scores, 0.25)
        self.assertEqual(threshold, 0.9)
        self.assertEqual(selective(scores, threshold)["accepted"], 20)

    def test_gate_does_not_claim_risk_when_nothing_is_accepted(self):
        result = selective([(0.1, False)], 1.01)
        self.assertEqual(result["coverage"], 0.0)
        self.assertIsNone(result["observed_missing_gold_risk"])

    def test_calibration_diagnostics_reward_exact_probabilities(self):
        exact = calibration_diagnostics([(1.0, True), (0.0, False)])
        uncertain = calibration_diagnostics([(0.5, True), (0.5, False)])
        self.assertEqual(exact["brier_score"], 0.0)
        self.assertGreater(uncertain["brier_score"], exact["brier_score"])

    def test_selective_interval_resamples_labels_as_clusters(self):
        scores = [(0.9, True, "label-a"), (0.9, True, "label-a"),
                  (0.9, False, "label-b"), (0.9, False, "label-b")]
        result = cluster_bootstrap_selective(scores, 0.8, draws=200, seed=7)
        self.assertEqual(result["n_labels"], 2)
        self.assertEqual(result["observed_missing_gold_risk"], 0.5)
        self.assertEqual(result["coverage_ci95_label_bootstrap"], [1.0, 1.0])
        self.assertEqual(result["missing_gold_risk_ci95_label_bootstrap"], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
