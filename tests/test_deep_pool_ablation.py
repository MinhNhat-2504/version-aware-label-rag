import unittest

from labelqa.data import Label
from labelqa.deep_pool_ablation import deep_feature_pool, deep_retrieve, paired_cells
from labelqa.data import Question
from labelqa.learned_retrieval import FEATURE_NAMES
from labelqa.retrieval import LabelIndex


class DeepPoolAblationTests(unittest.TestCase):
    def setUp(self):
        chunks = tuple(
            f"SECTION {index}\nSynthetic passage {index} alpha beta."
            for index in range(60)
        )
        self.index = LabelIndex(Label("synthetic", "Synthetic", chunks))

    def test_deep_pool_uses_requested_lexical_depth_without_duplicates(self):
        features, pool = deep_feature_pool(self.index, "alpha beta", pool_size=50)
        self.assertEqual(set(features), set(pool))
        self.assertEqual(len(pool), 50)
        self.assertEqual(len(pool), len(set(pool)))

    def test_deep_selector_is_deterministic_and_respects_budget(self):
        weights = [0.0] * len(FEATURE_NAMES)
        first = deep_retrieve(self.index, "alpha beta", weights, k=5, pool_size=50)
        second = deep_retrieve(self.index, "alpha beta", weights, k=5, pool_size=50)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)

    def test_invalid_pool_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            deep_feature_pool(self.index, "alpha", pool_size=0)

    def test_paired_cells_count_wins_and_losses(self):
        first = Question("q1", "s1", "multihop", "question", (1, 2))
        second = Question("q2", "s2", "multihop", "question", (1, 2))
        cells = paired_cells([(first, [1, 2]), (second, [1, 0])],
                             [(first, [1, 0]), (second, [1, 2])])
        self.assertEqual(cells["baseline_only"], 1)
        self.assertEqual(cells["deep_pool_only"], 1)


if __name__ == "__main__":
    unittest.main()
