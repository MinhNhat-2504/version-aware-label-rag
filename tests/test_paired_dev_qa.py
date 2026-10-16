import unittest

from labelqa.prepare_paired_dev_qa import select_rows
from labelqa.summarize_paired_dev_qa import paired_interval


class PairedDevelopmentQATests(unittest.TestCase):
    def test_paired_interval_uses_label_clusters(self):
        rows = [
            {"set_id": "a", "bm25": {"ok": False}, "learned_cover": {"ok": True}},
            {"set_id": "a", "bm25": {"ok": False}, "learned_cover": {"ok": True}},
            {"set_id": "b", "bm25": {"ok": True}, "learned_cover": {"ok": True}},
        ]
        result = paired_interval(rows, "ok", draws=200)
        self.assertEqual(result["n_labels"], 2)
        self.assertEqual(result["n_questions"], 3)
        self.assertEqual(result["candidate_minus_bm25"], 0.6667)

    def test_sample_keeps_labels_disjoint_and_retrieval_challenges_directional(self):
        rows = []
        for index in range(15):
            bm25 = [0, 1, 2, 3, 4] if index % 3 != 1 else [0, 2, 3, 4, 5]
            learned = [0, 1, 2, 3, 4] if index % 3 != 2 else [0, 2, 3, 4, 5]
            rows.append({"qid": str(index), "set_id": str(index), "gold_indices": [0, 1],
                         "rankings": {"bm25": bm25, "learned_cover": learned}})
        selected = select_rows(rows, representative_n=3, challenge_each=2)
        self.assertEqual(len(selected), 7)
        self.assertEqual(len({row["set_id"] for _, row in selected}), 7)
        for cohort, row in selected:
            gold = set(row["gold_indices"])
            baseline = gold <= set(row["rankings"]["bm25"])
            candidate = gold <= set(row["rankings"]["learned_cover"])
            if cohort == "retrieval_challenge_learned_win":
                self.assertTrue(candidate and not baseline)
            if cohort == "retrieval_challenge_bm25_win":
                self.assertTrue(baseline and not candidate)


if __name__ == "__main__":
    unittest.main()
