import unittest

from labelqa.summarize_holdout_retrieval import METHODS, paired_summary, summarize


class HoldoutRetrievalSummaryTests(unittest.TestCase):
    def test_pair_bootstrap_keeps_both_versions_together(self):
        items = [
            {"set_id": "a", "learned_cover": True, "bm25": False},
            {"set_id": "a", "learned_cover": True, "bm25": False},
            {"set_id": "b", "learned_cover": False, "bm25": True},
            {"set_id": "b", "learned_cover": False, "bm25": True},
        ]
        result = paired_summary(items, "learned_cover", "bm25", repetitions=1000)
        self.assertEqual(result["paired_cells"],
                         {"both": 0, "left_only": 2, "right_only": 2, "neither": 0})
        self.assertEqual(result["difference_percentage_points"], 0)
        self.assertEqual(result["pair_bootstrap_95ci_percentage_points"], [-100.0, 100.0])

    def test_pair_bootstrap_keeps_multiple_questions_with_their_label_pair(self):
        items = [{"set_id": "a", "learned_cover": True, "bm25": False} for _ in range(4)]
        items += [{"set_id": "b", "learned_cover": False, "bm25": True} for _ in range(2)]
        result = paired_summary(items, "learned_cover", "bm25", repetitions=1000)
        self.assertEqual(result["items"], 6)
        self.assertEqual(result["label_pairs"], 2)
        self.assertEqual(result["paired_cells"]["left_only"], 4)

    def test_complete_evidence_requires_each_claim_and_same_gold(self):
        row = {"qid": "q", "set_id": "a", "spl_version": 2, "question": "Q?",
               "xml_sha256": "x", "source_snapshot": "version_holdout_sources.jsonl",
               "answerable": True, "claims": [{"text": "one"}, {"text": "two"}]}
        prompts = {}
        for method, retrieved in (("bm25", [1, 2]), ("learned_cover", [1]),
                                  ("ontology_cover", [1, 2])):
            prompts[method] = {"q": {**{k: row[k] for k in
                                         ("qid", "set_id", "spl_version", "question",
                                          "xml_sha256", "source_snapshot")},
                                     "method": method, "task": "multihop", "gold_claims": [[1], [2]],
                                     "retrieved_indices": retrieved}}
        report = summarize([row], prompts)
        self.assertFalse(report["per_item"][0]["learned_cover"])
        self.assertTrue(report["per_item"][0]["bm25"])
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["answerable_items"], 1)
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["learned_vs_bm25"]["paired_cells"]["right_only"], 1)
        prompts["learned_cover"]["q"]["gold_claims"] = [[1], [3]]
        with self.assertRaisesRegex(ValueError, "Different accepted evidence groups"):
            summarize([row], prompts)

    def test_windows_snapshot_paths_match_their_posix_form(self):
        row = {"qid": "q", "set_id": "a", "spl_version": 2, "question": "Q?", "xml_sha256": "x",
               "source_snapshot": "data\\snapshots\\version_holdout_sources.jsonl",
               "answerable": True, "claims": [{"text": "one"}]}
        prompts = {method: {"q": {**{k: row[k] for k in ("qid", "set_id", "spl_version", "question", "xml_sha256")},
                                  "source_snapshot": "data/snapshots/version_holdout_sources.jsonl",
                                  "task": "factual", "gold_claims": [[1]], "retrieved_indices": [1]}}
                   for method in METHODS}
        self.assertTrue(summarize([row], prompts)["per_item"][0]["bm25"])
        prompts["bm25"]["q"]["source_snapshot"] = "data/snapshots/version_change_cohort_sources.jsonl"
        with self.assertRaisesRegex(ValueError, "source_snapshot"):
            summarize([row], prompts)

    def test_answerability_mismatch_fails(self):
        row = {"qid": "q", "set_id": "a", "spl_version": 2, "question": "Q?",
               "xml_sha256": "x", "source_snapshot": "version_holdout_sources.jsonl",
               "answerable": False, "claims": []}
        prompts = {method: {"q": {**{k: row[k] for k in
                                         ("qid", "set_id", "spl_version", "question",
                                          "xml_sha256", "source_snapshot")},
                                 "task": "factual"}} for method in METHODS}
        with self.assertRaisesRegex(ValueError, "answerability mismatch"):
            summarize([row], prompts)

    def test_unanswerable_item_still_enforces_five_passage_budget(self):
        row = {"qid": "q", "set_id": "a", "spl_version": 2, "question": "Q?",
               "xml_sha256": "x", "source_snapshot": "version_holdout_sources.jsonl",
               "answerable": False, "claims": []}
        prompts = {method: {"q": {**{k: row[k] for k in
                                         ("qid", "set_id", "spl_version", "question",
                                          "xml_sha256", "source_snapshot")},
                                 "task": "refusal", "retrieved_indices": list(range(6)),
                                 "gold_claims": []}} for method in METHODS}
        with self.assertRaisesRegex(ValueError, "Invalid passage budget"):
            summarize([row], prompts)


if __name__ == "__main__":
    unittest.main()
