import unittest

from labelqa.review_resolution import accepted_statuses, compiled_status, resolution


HUMANS = ["reader_a_01", "reader_b_01"]
MIXED = ["llm_annotator_claude", "expert_01"]


class ResolutionTests(unittest.TestCase):
    def test_existing_status_strings_are_unchanged(self):
        self.assertEqual(compiled_status("gold", "two_reader_consensus"),
                         "two_reader_consensus_compiled_no_automatic_agreement_to_gold")
        self.assertEqual(compiled_status("answer", "independent_adjudicator"),
                         "manual_answer_adjudication_compiled_no_automatic_agreement_to_gold")
        self.assertEqual(compiled_status("equivalence", "two_reader_consensus"),
                         "two_reader_equivalence_consensus_compiled_no_automatic_agreement_to_gold")
        self.assertIn(compiled_status("gold", "human_expert_resolution"), accepted_statuses("gold"))

    def test_human_reader_signs_expert_resolution_of_llm_annotation(self):
        mode, signers = resolution({"resolution_mode": "human_expert_resolution",
                                    "expert_resolver_id": "expert_01"}, MIXED, "q")
        self.assertEqual((mode, signers), ("human_expert_resolution", ["expert_01"]))

    def test_llm_cannot_sign_or_adjudicate(self):
        with self.assertRaisesRegex(ValueError, "human reader must sign"):
            resolution({"resolution_mode": "human_expert_resolution",
                        "expert_resolver_id": "llm_annotator_claude"}, MIXED, "q")
        with self.assertRaisesRegex(ValueError, "cannot sign a consensus"):
            resolution({"resolution_mode": "two_reader_consensus",
                        "consensus_reviewer_ids": MIXED}, MIXED, "q")
        with self.assertRaisesRegex(ValueError, "human adjudicator"):
            resolution({"resolution_mode": "independent_adjudicator",
                        "adjudicator_id": "llm_other"}, MIXED, "q")

    def test_expert_resolution_requires_exactly_one_llm_reader(self):
        with self.assertRaisesRegex(ValueError, "one LLM annotator"):
            resolution({"resolution_mode": "human_expert_resolution",
                        "expert_resolver_id": "reader_a_01"}, HUMANS, "q")
        with self.assertRaisesRegex(ValueError, "At most one"):
            resolution({"resolution_mode": "human_expert_resolution"},
                       ["llm_a", "llm_b"], "q")

    def test_compiled_record_uses_generic_signer_field(self):
        mode, signers = resolution({"resolution_mode": "human_expert_resolution",
                                    "resolver_ids": ["expert_01"]}, MIXED, "q")
        self.assertEqual(signers, ["expert_01"])


if __name__ == "__main__":
    unittest.main()
