import unittest

from labelqa.candidate_pool_diagnostics import failure_category, passage_count_bin


class CandidatePoolDiagnosticsTests(unittest.TestCase):
    def test_failure_categories_separate_generation_from_selection(self):
        gold = {1, 2}
        self.assertEqual(failure_category(gold, {0, 1, 2}, [1, 2]), "complete_top5")
        self.assertEqual(
            failure_category(gold, {0, 1, 2}, [0, 1]),
            "selection_miss_gold_available_in_pool",
        )
        self.assertEqual(failure_category(gold, {0, 1}, [0, 1]), "candidate_pool_miss")

    def test_passage_count_bins_cover_boundaries(self):
        self.assertEqual([passage_count_bin(value) for value in (25, 26, 50, 51, 100, 101)],
                         ["01-25", "26-50", "26-50", "51-100", "51-100", "101+"])


if __name__ == "__main__":
    unittest.main()
