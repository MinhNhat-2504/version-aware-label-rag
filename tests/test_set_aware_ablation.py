import unittest

from labelqa.data import Label
from labelqa.deep_pool_ablation import deep_feature_pool
from labelqa.learned_retrieval import FEATURE_NAMES
from labelqa.retrieval import LabelIndex
from labelqa.set_aware_ablation import (
    select_topk,
    structured_difference,
    train_set_aware,
)


class SetAwareAblationTests(unittest.TestCase):
    def setUp(self):
        self.index = LabelIndex(Label("synthetic", "Synthetic", (
            "INDICATIONS\nTreats Condition Alpha in adults.",
            "DOSAGE\nTake one unit daily.",
            "INTERACTIONS\nStop Drug Beta during treatment.",
            "STORAGE\nStore at room temperature.",
            "WARNINGS\nMonitor laboratory values.",
            "DESCRIPTION\nSynthetic inactive ingredients.",
        )))

    def test_structured_difference_moves_toward_missing_gold(self):
        positive = (1.0,) * len(FEATURE_NAMES)
        negative = (0.0,) * len(FEATURE_NAMES)
        difference = structured_difference({1: positive, 2: negative}, [1], [2])
        self.assertEqual(difference, [1.0] * len(FEATURE_NAMES))

    def test_set_training_and_selection_are_deterministic(self):
        question = "What treats Condition Alpha and what is required for Drug Beta?"
        features, pool = deep_feature_pool(self.index, question, pool_size=6)
        initial = [0.0] * len(FEATURE_NAMES)
        records = [(self.index, question, features, pool, {0, 2})]
        first, metadata = train_set_aware(records, initial)
        second, _ = train_set_aware(records, initial)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(FEATURE_NAMES))
        self.assertEqual(len(metadata["updates_by_epoch"]), 4)
        ranking = select_topk(self.index, question, features, pool, first, k=5)
        self.assertEqual(len(ranking), 5)


if __name__ == "__main__":
    unittest.main()
