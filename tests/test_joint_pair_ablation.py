import unittest

from labelqa.data import Label
from labelqa.deep_pool_ablation import deep_feature_pool
from labelqa.joint_pair_ablation import (
    PAIR_FEATURE_NAMES,
    joint_pair_retrieve,
    normalized_passage_scores,
    pair_features,
    train_pair_model,
)
from labelqa.learned_retrieval import FEATURE_NAMES
from labelqa.retrieval import LabelIndex


class JointPairAblationTests(unittest.TestCase):
    def setUp(self):
        self.index = LabelIndex(Label("synthetic", "Synthetic", (
            "INDICATIONS\nTreats Condition Alpha in adults.",
            "DOSAGE\nTake one unit daily.",
            "INTERACTIONS\nStop Drug Beta during treatment.",
            "STORAGE\nStore at room temperature.",
            "WARNINGS\nMonitor laboratory values.",
            "DESCRIPTION\nSynthetic inactive ingredients.",
        )))
        self.question = "What treats Condition Alpha and what is required for Drug Beta?"
        self.features, self.pool = deep_feature_pool(self.index, self.question, pool_size=6)
        self.passage_weights = [1.0] + [0.0] * (len(FEATURE_NAMES) - 1)

    def test_pair_features_are_symmetric_and_bounded(self):
        relevance = normalized_passage_scores(
            self.features, self.pool, self.passage_weights)
        forward = pair_features(
            self.index, self.question, self.features, relevance, 0, 2)
        reverse = pair_features(
            self.index, self.question, self.features, relevance, 2, 0)
        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), len(PAIR_FEATURE_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in forward))

    def test_training_and_retrieval_are_deterministic(self):
        records = [(self.index, self.question, self.features, self.pool, {0, 2})]
        first, metadata = train_pair_model(records, self.passage_weights)
        second, _ = train_pair_model(records, self.passage_weights)
        self.assertEqual(first, second)
        self.assertEqual(metadata["eligible_complete_pair_questions"], 1)
        ranking, selected_pair = joint_pair_retrieve(
            self.index, self.question, self.features, self.pool,
            self.passage_weights, first)
        self.assertEqual(len(ranking), 5)
        self.assertEqual(len(set(ranking)), 5)
        self.assertEqual(len(selected_pair), 2)

    def test_joint_retrieval_rejects_budget_below_pair_size(self):
        with self.assertRaises(ValueError):
            joint_pair_retrieve(
                self.index, self.question, self.features, self.pool,
                self.passage_weights, [0.0] * len(PAIR_FEATURE_NAMES), k=1)


if __name__ == "__main__":
    unittest.main()
