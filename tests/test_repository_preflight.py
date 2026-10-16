import unittest

from labelqa.export_reviewer_interface import ROOT
from labelqa.repository_preflight import check


class RepositoryPreflightTests(unittest.TestCase):
    def test_source_only_remote_is_ready(self):
        report = check("source", ROOT)
        self.assertEqual(report["status"], "pass", report["blockers"])
        self.assertTrue(report["checks"]["required_repository_files_present"])

    def test_public_release_remains_blocked_before_human_holdout(self):
        report = check("public", ROOT)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["manual_gold_and_receipt_present"])
        self.assertFalse(report["checks"]["holdout_execution_lock_present"])
        self.assertFalse(report["checks"]["holdout_answer_scores_present"])


if __name__ == "__main__":
    unittest.main()
