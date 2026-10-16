import unittest

from labelqa.prepare_blinded_answer_review import review_packet
from labelqa.summarize_holdout_retrieval import METHODS


class BlindedAnswerReviewTests(unittest.TestCase):
    def fixture(self):
        gold = [{"qid": "q1", "set_id": "label-1", "spl_version": 2,
                 "xml_sha256": "source-hash", "source_snapshot": "pinned-source.jsonl",
                 "question": "What does the label say?", "reference_answer": "secret gold"}]
        prompts = {method: {"q1": {"qid": "q1", "set_id": "label-1", "spl_version": 2,
                                   "xml_sha256": "source-hash", "question": "What does the label say?",
                                   "user_prompt": "Question...\nLabel passages:\n[P0001] source text"}}
                   for method in METHODS}
        responses = {method: {"q1": {"prompt_sha256": "hash", "model_digest": "model-hash",
                                     "decoding_options": {"temperature": 0},
                                     "parsed_response": {"answer": "candidate answer"},
                                     "raw_response": '{"answer":"x"}'}}
                     for method in METHODS}
        return gold, prompts, responses

    def test_packet_hides_method_and_gold_but_retains_private_mapping(self):
        gold, prompts, responses = self.fixture()
        serial = iter((f"blind-review-{i:02d}" for i in range(3)))
        packet, key = review_packet(gold, prompts, responses, id_factory=lambda: next(serial))
        self.assertEqual(len(packet), 3)
        self.assertEqual({x["method"] for x in key}, set(METHODS))
        self.assertEqual({x["review_id"] for x in packet}, {x["review_id"] for x in key})
        self.assertTrue(all("method" not in row and "reference_answer" not in row
                            and "answerable" not in row for row in packet))
        self.assertTrue(all(row["reviewer_id"] == "" and row["answer_complete"] is None
                            for row in packet))

    def test_rejects_different_answering_model(self):
        gold, prompts, responses = self.fixture()
        responses["bm25"]["q1"]["model_digest"] = "other-model"
        with self.assertRaisesRegex(ValueError, "model or decoding options differ"):
            review_packet(gold, prompts, responses)


if __name__ == "__main__":
    unittest.main()
