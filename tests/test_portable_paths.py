from pathlib import Path
import unittest

from labelqa.data import portable_path
from labelqa.holdout_reporting import cohort_for


class PortablePathTests(unittest.TestCase):
    def test_windows_manifest_path_works_on_every_platform(self):
        self.assertEqual(
            portable_path(r"data\snapshots\version_holdout_sources.jsonl"),
            Path("data/snapshots/version_holdout_sources.jsonl"),
        )

    def test_cohort_lookup_accepts_windows_separator(self):
        self.assertEqual(
            cohort_for(r"data\snapshots\version_holdout_sources.jsonl"),
            "consecutive_pairs",
        )


if __name__ == "__main__":
    unittest.main()
