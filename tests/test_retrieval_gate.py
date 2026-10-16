import unittest

from labelqa.calibrate_retrieval_gate import choose_threshold, selective


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


if __name__ == "__main__":
    unittest.main()
