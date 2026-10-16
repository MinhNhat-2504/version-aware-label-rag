import unittest

from labelqa.synthetic_demo import run_demo


class SyntheticDemoTests(unittest.TestCase):
    def test_smoke_demo_exercises_table_retrieval_and_citation_scoring(self):
        report = run_demo()
        self.assertEqual(report["status"], "synthetic_smoke_passed")
        self.assertFalse(report["contains_real_drug_or_holdout_data"])
        self.assertFalse(report["model_called"])
        self.assertEqual(report["parser"]["table_row_count"], 1)
        self.assertTrue(report["retrieval"]["evidence_set"]["complete_evidence"])
        self.assertTrue(report["mechanical_score"]["valid_schema"])
        self.assertTrue(report["mechanical_score"]["cited_complete_evidence"])
        self.assertEqual(report["mechanical_score"]["invalid_citation_count"], 0)


if __name__ == "__main__":
    unittest.main()
