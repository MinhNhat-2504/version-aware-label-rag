from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import unittest

from labelqa.check_holdout_reviews import check
from labelqa.compile_holdout_gold import compile_gold


class HoldoutReviewTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {"qid": "q", "pair_id": "sample", "set_id": "sample", "spl_version": 2,
                          "xml_sha256": "pinned", "question": "What fact?",
                          "source_snapshot": "source.jsonl"}
        self.review = {**self.candidate, "item_decision": "accept", "independent_answer": "Fact.",
                       "answerable_from_version": True, "required_claims": ["Fact"],
                       "acceptable_section_indices_per_claim": [[7]],
                       "acceptable_evidence_spans_per_claim": [[
                           {"section_index": 7, "source_kind": "prose", "quote": "Fact."}]],
                       "version_change_judgment": "stable"}
        self.a = {**self.review, "reviewer_id": "reader-a"}
        self.b = {**self.review, "reviewer_id": "reader-b"}
        self.source = {"set_id": "sample", "versions": [{"spl_version": 2,
                       "xml_sha256": "pinned", "sections": [{"section_index": 7,
                       "title": "Indications", "text": "Fact."}]}]}

    def run_check(self):
        paths = [Path("candidate"), Path("a"), Path("b")]
        records = {paths[0]: {"q": self.candidate}, paths[1]: {"q": self.a},
                   paths[2]: {"q": self.b}}
        with patch("labelqa.check_holdout_reviews.index_reviews", side_effect=lambda p: records[p]):
            with patch("labelqa.check_holdout_reviews.read_jsonl", return_value=iter([self.source])):
                return check(*paths)

    def test_distinct_reviews_still_require_manual_adjudication(self):
        report = self.run_check()
        self.assertTrue(report["all_items_require_manual_adjudication"])
        self.assertEqual(report["queue"][0]["status"], "needs_adjudication")

    def test_same_reviewer_is_rejected(self):
        self.b["reviewer_id"] = "reader-a"
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.run_check()

    def test_invalid_source_section_is_rejected(self):
        self.b["acceptable_section_indices_per_claim"] = [[8]]
        with self.assertRaisesRegex(ValueError, "absent from pinned source"):
            self.run_check()

    def test_quote_must_come_from_pinned_section(self):
        self.b["acceptable_evidence_spans_per_claim"][0][0]["quote"] = "Invented text"
        with self.assertRaisesRegex(ValueError, "quote absent"):
            self.run_check()

    def test_table_span_requires_row_locator(self):
        self.b["acceptable_evidence_spans_per_claim"][0][0]["source_kind"] = "table_row"
        with self.assertRaisesRegex(ValueError, "row locator"):
            self.run_check()

    def test_matching_reviews_do_not_become_gold_automatically(self):
        queue = self.run_check()
        with self.assertRaisesRegex(ValueError, "resolution mode"):
            compile_gold(queue, queue, {"q": self.candidate}, {"q": {7}})

    def test_two_reader_consensus_can_compile_gold(self):
        expected = self.run_check()
        edited = deepcopy(expected)
        edited["queue"][0].update({
            "resolution_mode": "two_reader_consensus",
            "consensus_reviewer_ids": ["reader-a", "reader-b"],
            "adjudicator_id": "", "final_decision": "accept",
            "final_answerable_from_version": True, "final_answer": "Fact.",
            "final_claims": ["Fact"], "final_acceptable_section_indices_per_claim": [[7]],
            "final_acceptable_evidence_spans_per_claim": [[
                {"section_index": 7, "source_kind": "prose", "quote": "Fact."}]],
            "final_version_change_judgment": "stable",
            "adjudication_notes": "Both readers checked and signed the consensus."})
        accepted, excluded = compile_gold(
            expected, edited, {"q": self.candidate}, {"q": {7: "Fact."}})
        self.assertEqual(excluded, [])
        self.assertEqual(accepted[0]["resolution_mode"], "two_reader_consensus")
        self.assertEqual(accepted[0]["resolver_ids"], ["reader-a", "reader-b"])
        invalid = deepcopy(edited)
        invalid["queue"][0]["consensus_reviewer_ids"] = ["reader-a", "reader-a"]
        with self.assertRaisesRegex(ValueError, "Both original readers"):
            compile_gold(expected, invalid, {"q": self.candidate}, {"q": {7: "Fact."}})

    def test_manual_gold_requires_valid_final_section(self):
        expected = self.run_check()
        edited = deepcopy(expected)
        row = edited["queue"][0]
        row.update({"adjudicator_id": "expert-c", "final_decision": "accept",
                    "final_answerable_from_version": True, "final_answer": "Fact.",
                    "final_claims": ["Fact"], "final_acceptable_section_indices_per_claim": [[8]],
                    "final_acceptable_evidence_spans_per_claim": [[
                        {"section_index": 7, "source_kind": "prose", "quote": "Fact."}]],
                    "final_version_change_judgment": "stable", "adjudication_notes": "Checked source."})
        with self.assertRaisesRegex(ValueError, "absent section"):
            compile_gold(expected, edited, {"q": self.candidate}, {"q": {7: "Fact."}})
        row["final_acceptable_section_indices_per_claim"] = [[7]]
        accepted, excluded = compile_gold(expected, edited, {"q": self.candidate}, {"q": {7: "Fact."}})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(excluded, [])

    def test_rewrite_requires_new_review_round(self):
        expected = self.run_check()
        edited = deepcopy(expected)
        edited["queue"][0].update({"adjudicator_id": "expert-c", "final_decision": "rewrite",
                                    "adjudication_notes": "Question wording changed."})
        with self.assertRaisesRegex(ValueError, "new frozen candidate"):
            compile_gold(expected, edited, {"q": self.candidate}, {"q": {7: "Fact."}})

    def test_compiler_rejects_changed_reader_evidence(self):
        expected = self.run_check()
        edited = deepcopy(expected)
        edited["queue"][0]["reader_a"]["independent_answer"] = "Edited after review"
        with self.assertRaisesRegex(ValueError, "Reader evidence"):
            compile_gold(expected, edited, {"q": self.candidate}, {"q": {7: "Fact."}})


if __name__ == "__main__":
    unittest.main()
