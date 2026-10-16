from copy import deepcopy
import unittest

from labelqa.answer_review import FIELDS, check, compile_final, mechanical, packet_index, validate_review
from labelqa.prepare_blinded_answer_review import review_packet
from labelqa.summarize_answer_reviews import summarize
from labelqa.summarize_holdout_retrieval import METHODS


class AnswerReviewPipelineTests(unittest.TestCase):
    def fixture(self):
        gold = [
            {"qid": "q1", "set_id": "pair-1", "spl_version": 1,
             "xml_sha256": "hash-1", "source_snapshot": "version_holdout_sources.jsonl",
             "question": "Is it indicated?", "answerable": True},
            {"qid": "q2", "set_id": "pair-1", "spl_version": 2,
             "xml_sha256": "hash-2", "source_snapshot": "version_holdout_sources.jsonl",
             "question": "Is interaction stated?", "answerable": False},
        ]
        prompts = {method: {} for method in METHODS}
        responses = {method: {} for method in METHODS}
        for method in METHODS:
            for row in gold:
                qid = row["qid"]
                prompts[method][qid] = {**{k: row[k] for k in
                                           ("qid", "set_id", "spl_version", "xml_sha256", "question")},
                                        "user_prompt": "Question\nLabel passages:\n[P0000] Synthetic text"}
                parsed = ({"answer": "Not stated.", "abstain": True, "claims": []}
                          if qid == "q2" and method != "learned_cover" else
                          {"answer": "Synthetic answer.", "abstain": False,
                           "claims": [{"text": "Synthetic claim", "citations": ["P0000"]}]})
                if qid == "q1" and method == "ontology_cover":
                    parsed = None
                responses[method][qid] = {"prompt_sha256": "synthetic-prompt-hash",
                                          "model_digest": "synthetic-model",
                                          "decoding_options": {"temperature": 0},
                                          "parsed_response": parsed,
                                          "raw_response": "synthetic raw output",
                                          "error": "invalid JSON" if parsed is None else None}
        serial = iter(f"blind-review-{i:02d}" for i in range(6))
        packet, mapping = review_packet(gold, prompts, responses, id_factory=lambda: next(serial))
        return gold, packet, mapping

    @staticmethod
    def filled(packet, reviewer_id, supported=True):
        review = deepcopy(packet)
        review["reviewer_id"] = reviewer_id
        review["review_notes"] = "Checked the pinned synthetic source."
        machine = mechanical(packet)
        if not machine["valid_schema"]:
            review["review_status"] = "invalid_output"
        elif machine["abstain"]:
            review.update({"review_status": "assessed", "correct_version": True,
                           "unsafe_unsupported_claim": False, "refusal_appropriate": True})
        else:
            review.update({"review_status": "assessed", "answer_complete": True,
                           "all_claims_supported": supported,
                           "citations_at_correct_location": True, "correct_version": True,
                           "unsafe_unsupported_claim": not supported,
                           "claim_judgments": [{"claim_index": 0,
                                                "support": "supported" if supported else "unsupported",
                                                "citation_location_valid": True,
                                                "notes": "" if supported else "Unsupported relation."}]})
        return review

    def test_two_readers_manual_reconciliation_and_final_scoring(self):
        gold, packet, mapping = self.fixture()
        lookup = {x["review_id"]: x for x in mapping}
        first = {row["review_id"]: self.filled(row, "reader-a",
                 supported=not (row["qid"] == "q1" and lookup[row["review_id"]]["method"] == "learned_cover"))
                 for row in packet}
        second = {row["review_id"]: self.filled(row, "reader-b",
                  supported=not (row["qid"] == "q1" and lookup[row["review_id"]]["method"] in
                                 ("learned_cover", "bm25"))) for row in packet}
        queue = check(packet, first, second, ("reader-a", "reader-b"))
        self.assertEqual(queue["items"], 6)
        self.assertTrue(any("all_claims_supported_differs" in row["reasons"] for row in queue["queue"]))
        with self.assertRaisesRegex(ValueError, "resolution mode"):
            compile_final(queue, queue, packet_index(packet))
        edited = deepcopy(queue)
        for row in edited["queue"]:
            row["adjudicator_id"] = "expert-c"
            row["adjudication_notes"] = "Resolved independently on synthetic source."
            for field in ("review_status", "answer_complete", "all_claims_supported",
                          "citations_at_correct_location", "correct_version", "refusal_appropriate",
                          "unsafe_unsupported_claim", "claim_judgments", "review_notes"):
                row["final_" + field] = row["reader_a"][field]
        final = compile_final(queue, edited, packet_index(packet))
        consensus = deepcopy(edited)
        for row in consensus["queue"]:
            row.update({"resolution_mode": "two_reader_consensus",
                        "consensus_reviewer_ids": ["reader-a", "reader-b"],
                        "adjudicator_id": ""})
        consensus_final = compile_final(queue, consensus, packet_index(packet))
        self.assertTrue(all(row["resolution_mode"] == "two_reader_consensus"
                            and row["resolver_ids"] == ["reader-a", "reader-b"]
                            for row in consensus_final))
        report = summarize(gold, packet, {"mapping": mapping}, consensus_final)
        self.assertEqual(report["method_counts"]["bm25"]["end_to_end_successes"], 2)
        self.assertEqual(report["method_counts"]["learned_cover"]["end_to_end_successes"], 0)
        self.assertEqual(report["method_counts"]["ontology_cover"]["end_to_end_successes"], 1)
        self.assertEqual(report["method_counts"]["ontology_cover"]["invalid_or_unusable_outputs"], 1)
        self.assertEqual(report["paired_all_accepted"]["learned_vs_bm25"]["paired_cells"]["right_only"], 2)
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["accepted_items"], 2)
        self.assertEqual(report["by_cohort"]["consecutive_pairs"]["method_counts"]["bm25"]["end_to_end_successes"], 2)
        self.assertEqual(report["by_cohort"]["text_change_enriched"]["accepted_items"], 0)
        edited["queue"][0]["reader_a"]["question"] = "Tampered question"
        with self.assertRaisesRegex(ValueError, "Original answer reviews"):
            compile_final(queue, edited, packet_index(packet))

    def test_reviewer_cannot_change_answer_or_skip_claim_judgment(self):
        _, packet, _ = self.fixture()
        answer = next(row for row in packet if mechanical(row)["valid_schema"] and
                      mechanical(row)["abstain"] is False)
        review = self.filled(answer, "reader-a")
        review["model_answer"]["answer"] = "Changed"
        with self.assertRaisesRegex(ValueError, "Frozen answer/source fields changed"):
            validate_review(answer, review)
        review = self.filled(answer, "reader-a")
        review["claim_judgments"] = []
        with self.assertRaisesRegex(ValueError, "One judgment is required"):
            validate_review(answer, review)

    def test_invalid_citation_blocks_end_to_end_success_even_if_reader_marks_support(self):
        gold, packet, mapping = self.fixture()
        for row in packet:
            if row["qid"] == "q1" and next(x["method"] for x in mapping
                                            if x["review_id"] == row["review_id"]) == "bm25":
                row["model_answer"]["claims"][0]["citations"] = ["P9999"]
                self.assertEqual(mechanical(row)["invalid_citation_count"], 1)
                break
        final = []
        for row in packet:
            review = self.filled(row, "expert-c")
            final.append({"review_id": row["review_id"], "qid": row["qid"],
                          "adjudicator_id": "expert-c", "adjudication_notes": "Synthetic source checked.",
                          **{key: review[key] for key in FIELDS}})
        report = summarize(gold, packet, {"mapping": mapping}, final)
        bm25_q1 = next(row for row in report["per_item"]
                       if row["method"] == "bm25" and row["qid"] == "q1")
        self.assertFalse(bm25_q1["end_to_end_success"])


if __name__ == "__main__":
    unittest.main()
