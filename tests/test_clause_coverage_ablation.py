import unittest

from labelqa.clause_coverage_ablation import clause_retrieve, select_weight
from labelqa.data import Label
from labelqa.learned_retrieval import FEATURE_NAMES
from labelqa.retrieval import LabelIndex


class ClauseCoverageAblationTests(unittest.TestCase):
    def setUp(self):
        self.index = LabelIndex(Label("synthetic", "Synthetic", (
            "INDICATIONS\nTreats Condition Alpha in adults.",
            "WARNINGS\nMonitor laboratory values during treatment.",
            "INTERACTIONS\nDrug Beta must be stopped during treatment.",
        )))
        self.weights = [0.0] * len(FEATURE_NAMES)

    def test_clause_selector_is_deterministic_and_respects_budget(self):
        question = "What treats Condition Alpha and what is required for Drug Beta?"
        first = clause_retrieve(self.index, question, self.weights, 0.2, k=2)
        second = clause_retrieve(self.index, question, self.weights, 0.2, k=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first)), 2)

    def test_weight_selection_prefers_zero_when_all_candidates_tie(self):
        selected, counts = select_weight(
            [(self.index, "What treats Condition Alpha?", {0})],
            self.weights,
            grid=(0.0, 0.2),
        )
        self.assertEqual(selected, 0.0)
        self.assertEqual(counts["0.0"], counts["0.2"])

    def test_negative_clause_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            clause_retrieve(self.index, "Question", self.weights, -0.1)


if __name__ == "__main__":
    unittest.main()
