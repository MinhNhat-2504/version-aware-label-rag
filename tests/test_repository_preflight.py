import json
import shutil
import unittest
from unittest.mock import patch

from labelqa.export_reviewer_interface import ROOT
from labelqa.repository_preflight import check, holdout_gates, prohibited_source_paths


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

    def test_public_release_remains_blocked_before_holdout_execution(self):
        if (ROOT / "paper/HOLDOUT_EXECUTION_LOCK.json").is_file():
            self.skipTest("This workspace has already run the locked holdout")
        report = check("public", ROOT)
        self.assertEqual(report["status"], "blocked")
        if not (ROOT / "data/holdout/version_silver_ai.jsonl").is_file():
            self.assertFalse(report["checks"]["holdout_reference_valid"])
        self.assertFalse(report["checks"]["holdout_execution_lock_valid"])
        self.assertFalse(report["checks"]["holdout_answer_endpoint_matches_reference_and_lock"])

    def test_public_gates_verify_reports_against_gold_and_lock(self):
        root = ROOT / "results" / ".test-public-gates"
        shutil.rmtree(root, ignore_errors=True)
        (root / "results").mkdir(parents=True)
        try:
            (root / "results/holdout_retrieval_summary.json").write_text(json.dumps(
                {"status": "after_manual_gold_retrieval_only_no_answer_support_claim",
                 "gold_sha256": "gold", "execution_lock_sha256": "lock"}), encoding="utf-8")
            (root / "results/holdout_answer_scores.json").write_text(json.dumps(
                {"status": "manual_adjudicated_end_to_end_answer_score_after_gold",
                 "gold_sha256": "other-gold", "execution_lock_sha256": "lock"}), encoding="utf-8")
            with (patch("labelqa.prepare_holdout_eval.load_manual_gold",
                        return_value=([], {"gold_sha256": "gold"})),
                  patch("labelqa.lock_holdout_execution.verify_lock", return_value=({}, "lock"))):
                gates, reasons = holdout_gates(root)
        finally:
            shutil.rmtree(root, ignore_errors=True)
        self.assertTrue(gates["holdout_execution_lock_valid"])
        self.assertTrue(gates["holdout_retrieval_summary_matches_gold_and_lock"])
        self.assertFalse(gates["holdout_answer_endpoint_matches_reference_and_lock"])
        self.assertEqual(len(reasons), 1)


if __name__ == "__main__":
    unittest.main()
