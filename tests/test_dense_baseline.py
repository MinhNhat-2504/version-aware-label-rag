import unittest

from labelqa.dense_baseline import normalized, rank_dense, rank_rrf


class DenseBaselineTests(unittest.TestCase):
    def test_cosine_ranking_and_stable_ties(self):
        query = normalized([3.0, 4.0])
        chunks = [normalized([0.0, 1.0]), normalized([1.0, 0.0]),
                  normalized([3.0, 4.0]), normalized([3.0, 4.0])]
        self.assertEqual(rank_dense(query, chunks), [2, 3, 0, 1])
        with self.assertRaises(ValueError):
            rank_dense(query, [[1.0]])

    def test_rrf_equal_rankings_and_mismatched_sets(self):
        self.assertEqual(rank_rrf([2, 0, 1], [2, 0, 1]), [2, 0, 1])
        self.assertEqual(rank_rrf([0, 1, 2], [2, 1, 0]), [0, 2, 1])
        with self.assertRaises(ValueError):
            rank_rrf([0, 1], [0, 2])


if __name__ == "__main__":
    unittest.main()
