import unittest

from labelqa.holdout_reporting import sample_flow


class HoldoutReportingTests(unittest.TestCase):
    def test_rejected_version_is_reported_without_counting_as_model_failure(self):
        primary = "data/snapshots/version_holdout_sources.jsonl"
        enriched = "data/snapshots/version_change_cohort_sources.jsonl"
        gold = [
            {"qid": "p1-old", "set_id": "pair-1", "spl_version": 1,
             "source_snapshot": primary, "answerable": True},
            {"qid": "p2-old", "set_id": "pair-2", "spl_version": 3,
             "source_snapshot": enriched, "answerable": False},
            {"qid": "p2-new", "set_id": "pair-2", "spl_version": 4,
             "source_snapshot": enriched, "answerable": True},
        ]
        receipt = {"candidate_items": 4, "accepted": 3, "excluded": 1,
                   "exclusions": [{"qid": "p1-new", "set_id": "pair-1", "spl_version": 2,
                                   "source_snapshot": primary, "reason": "Question cannot be answered reliably.",
                                   "adjudicator_id": "expert-c"}]}
        report = sample_flow(gold, receipt)
        self.assertEqual(report["candidate_pairs"], 2)
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["one_version_accepted"], 1)
        self.assertEqual(report["by_cohort"]["text_change_enriched"]["both_versions_accepted"], 1)
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["excluded_items"], 1)
        receipt["candidate_items"] = 5
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            sample_flow(gold, receipt)

    def test_multiple_questions_per_version_are_counted_within_the_same_pair(self):
        source = "version_holdout_sources.jsonl"
        gold = [{"qid": "first", "set_id": "pair", "spl_version": 1,
                 "source_snapshot": source, "answerable": True},
                {"qid": "second", "set_id": "pair", "spl_version": 1,
                 "source_snapshot": source, "answerable": False},
                {"qid": "third", "set_id": "pair", "spl_version": 2,
                 "source_snapshot": source, "answerable": True}]
        receipt = {"candidate_items": 3, "accepted": 3, "excluded": 0, "exclusions": []}
        report = sample_flow(gold, receipt)
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["both_versions_accepted"], 1)
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["candidate_items"], 3)
        gold.pop()
        receipt.update({"candidate_items": 2, "accepted": 2})
        with self.assertRaisesRegex(ValueError, "two SPL versions"):
            sample_flow(gold, receipt)


if __name__ == "__main__":
    unittest.main()
