import unittest

from labelqa.export_data_archive import GROUPS


class DataArchiveTests(unittest.TestCase):
    def test_archive_never_copies_drafts_bundles_or_the_benchmark(self):
        patterns = [pattern for group in GROUPS.values() for pattern in group]
        for forbidden in ("KHONG_DUNG", "review-data", "answer-review-data", "guides_backup",
                          "data/upstream", "private_map", "answer_key"):
            self.assertFalse(any(forbidden in pattern for pattern in patterns), forbidden)
        self.assertIn("data/holdout/version_silver_ai.jsonl", GROUPS["holdout/silver_standard"])


if __name__ == "__main__":
    unittest.main()
