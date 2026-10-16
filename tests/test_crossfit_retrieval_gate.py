import unittest

from labelqa.crossfit_retrieval_gate import (
    clustered_upper_risk_bounds,
    crossfit_scores,
    fixed_sequence_threshold,
    threshold_grid,
)


def labelled_scores(rows):
    """Expand (score, correct, label, copies) into gate score tuples."""
    return [(score, correct, label) for score, correct, label, copies in rows
            for _ in range(copies)]


class CrossfitRetrievalGateTests(unittest.TestCase):
    def test_out_of_fold_scores_never_use_the_scored_fold(self):
        # The feature predicts success in fold 0 and failure in fold 1, so only an
        # in-sample fit would score fold-0 positives above one half.
        examples = []
        for fold, positive_means_success in ((0, True), (1, False)):
            for i in range(20):
                feature = float(i % 2)
                correct = bool(feature) == positive_means_success
                examples.append((f"q{fold}-{i}", f"label-{fold}-{i % 5}", fold, (feature,), correct))
        scores = crossfit_scores(examples)
        self.assertEqual(len(scores), len(examples))
        fold0_positive = [score for score, row in zip(scores, examples)
                          if row[2] == 0 and row[3] == (1.0,)]
        fold0_negative = [score for score, row in zip(scores, examples)
                          if row[2] == 0 and row[3] == (0.0,)]
        self.assertLess(max(fold0_positive), min(fold0_negative))
        with self.assertRaises(ValueError):
            crossfit_scores([(q, label, 0, values, correct)
                             for q, label, _, values, correct in examples])

    def test_threshold_grid_is_ordered_from_conservative_to_permissive(self):
        grid = threshold_grid([0.9, 0.8, 0.7, 0.1], coverages=(0.25, 0.5, 1.0))
        self.assertEqual(grid, [0.9, 0.8, 0.1])

    def test_upper_bound_respects_label_clusters(self):
        scores = labelled_scores([(0.9, True, "a", 10), (0.9, False, "b", 10)])
        bound = clustered_upper_risk_bounds(scores, [0.5], draws=400, seed=3)[0]
        # Resampling two labels gives risk 0, 0.5 or 1; the one-sided bound is 1.
        self.assertEqual(bound, 1.0)

    def test_fixed_sequence_stops_at_first_failure(self):
        safe = [(0.9, True, f"safe-{i}", 6) for i in range(20)]
        risky = [(0.4, False, f"risky-{i}", 6) for i in range(20)]
        scores = labelled_scores(safe + risky)
        threshold, trace = fixed_sequence_threshold(scores, 0.25, min_accepted=10,
                                                    draws=300, seed=5)
        self.assertEqual(threshold, 0.9)
        self.assertTrue(trace[0]["passes"])
        self.assertFalse(trace[-1]["passes"])
        self.assertEqual(sum(not row["passes"] for row in trace), 1)

    def test_gate_abstains_on_all_when_first_test_fails(self):
        scores = labelled_scores([(0.9, False, f"label-{i}", 5) for i in range(20)])
        threshold, trace = fixed_sequence_threshold(scores, 0.25, min_accepted=10,
                                                    draws=200, seed=7)
        self.assertEqual(threshold, 1.01)
        self.assertEqual(len(trace), 1)


if __name__ == "__main__":
    unittest.main()
