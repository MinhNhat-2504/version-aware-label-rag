"""Disclosed LLM answer review plus one human expert who resolves every item."""

from copy import deepcopy
import unittest

from labelqa.answer_review import FIELDS, check, compile_final, packet_index
from labelqa.build_llm_answer_review import build_rows
from labelqa.expert_resolution import sign
from labelqa.review_resolution import review_design
from labelqa.summarize_answer_reviews import limitations, summarize

import test_answer_review_pipeline as base  # Module import: its tests are not collected twice.


LLM, EXPERT = "llm_answer_reviewer_test", "expert_01"


class LlmExpertAnswerReviewTests(unittest.TestCase):
    def setUp(self):
        self.gold, self.packet, self.mapping = base.AnswerReviewPipelineTests().fixture()

    def llm_spec(self):
        items = {}
        for row in self.packet:
            filled = base.AnswerReviewPipelineTests.filled(row, LLM)
            items[row["review_id"]] = {field: filled[field] for field in FIELDS}
        return {"annotator_id": LLM, "model": "test", "disclosures": ["Synthetic LLM reviewer."],
                "items": items}

    def test_llm_spec_becomes_a_validated_reader_file(self):
        rows = build_rows(self.llm_spec(), self.packet)
        self.assertEqual({row["reviewer_id"] for row in rows}, {LLM})
        bad = self.llm_spec()
        bad["annotator_id"] = "reader_x"
        with self.assertRaisesRegex(ValueError, "llm_"):
            build_rows(bad, self.packet)
        missing = self.llm_spec()
        missing["items"].pop(self.packet[0]["review_id"])
        with self.assertRaisesRegex(ValueError, "exactly the packet"):
            build_rows(missing, self.packet)
        invalid = self.llm_spec()
        invalid["items"][self.packet[0]["review_id"]]["review_notes"] = ""
        with self.assertRaisesRegex(ValueError, "notes"):
            build_rows(invalid, self.packet)

    def test_expert_signs_and_scores_with_expert_llm_limitation(self):
        llm = {row["review_id"]: row for row in build_rows(self.llm_spec(), self.packet)}
        expert = {row["review_id"]: base.AnswerReviewPipelineTests.filled(row, EXPERT) for row in self.packet}
        queue = check(self.packet, expert, llm, (EXPERT, LLM))
        sheet = {row["review_id"]: {"choice": "expert", "notes": "Kept my blind review."}
                 for row in self.packet}
        signed = sign(queue, sheet, EXPERT)
        final = compile_final(queue, signed, packet_index(self.packet))
        self.assertEqual({tuple(row["resolver_ids"]) for row in final}, {(EXPERT,)})
        report = summarize(self.gold, self.packet, {"mapping": self.mapping}, final)
        self.assertIn("accepted_items", report)
        design = review_design({"reader_ids": [EXPERT, LLM], "resolution_mode": "human_expert_resolution",
                                "resolver_ids": [EXPERT]})
        self.assertEqual(design["llm_annotator_ids"], [LLM])
        self.assertIn("expert-LLM agreement", limitations(design)[0])
        # The LLM can never be the signer, even when its review is adopted.
        forged = deepcopy(signed)
        forged["queue"][0]["expert_resolver_id"] = LLM
        with self.assertRaisesRegex(ValueError, "human reader must sign"):
            compile_final(queue, forged, packet_index(self.packet))


if __name__ == "__main__":
    unittest.main()
