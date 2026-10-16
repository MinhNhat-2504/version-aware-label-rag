import unittest
from unittest.mock import patch

from labelqa.export_reviewer_interface import ROOT
from labelqa.repository_preflight import check, prohibited_source_paths


class RepositoryPreflightTests(unittest.TestCase):
    def test_source_only_remote_is_ready(self):
        report = check("source", ROOT)
        self.assertEqual(report["status"], "pass", report["blockers"])
        self.assertTrue(report["checks"]["required_repository_files_present"])
        self.assertTrue(report["checks"]["source_tree_excludes_private_artifacts"])
        self.assertNotIn("working_manifest_declares_pending_human_review", report["checks"])

    def test_source_only_remote_rejects_private_tracked_artifacts(self):
        tracked = ["README.md", "src/labelqa/data.py", "paper/MANUSCRIPT_DRAFT.docx"]
        with patch("labelqa.repository_preflight.source_tree_paths",
                   return_value=(tracked, "git_index")):
            report = check("source", ROOT)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["source_tree_excludes_private_artifacts"])
        self.assertIn("paper/MANUSCRIPT_DRAFT.docx", report["blockers"][0])

    def test_prohibited_source_paths_normalizes_windows_and_case(self):
        paths = [
            "src/labelqa/data.py",
            "data\\holdout\\gold.jsonl",
            "PAPER/Manuscript.md",
            "reviewer_ui/dist/review-data.js",
            "submission.DOCX",
            "source-notes.md",
        ]
        self.assertEqual(
            prohibited_source_paths(paths),
            [
                "PAPER/Manuscript.md",
                "data/holdout/gold.jsonl",
                "reviewer_ui/dist/review-data.js",
                "submission.DOCX",
            ],
        )

    def test_public_release_remains_blocked_before_human_holdout(self):
        report = check("public", ROOT)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["checks"]["manual_gold_and_receipt_present"])
        self.assertFalse(report["checks"]["holdout_execution_lock_present"])
        self.assertFalse(report["checks"]["holdout_answer_scores_present"])


if __name__ == "__main__":
    unittest.main()
