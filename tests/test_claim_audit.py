import unittest

from labelqa.audit_claims import audit_one, numeric_units, passage_blocks


class ClaimAuditTests(unittest.TestCase):
    def test_literal_numeric_unit_mismatch_is_flagged(self):
        prompt = {"qid": "q", "user_prompt": "Question\n\n[P0001]\nChildren from 2 years."}
        response = {"parsed_response": {"answer": "Children from 4 years.", "abstain": False,
                                        "claims": [{"text": "Children from 2 years.",
                                                    "citations": ["P0001"]}]}}
        kinds = {flag["kind"] for flag in audit_one(prompt, response)["flags"]}
        self.assertEqual(kinds, {"answer_numeric_unit_absent_from_claims"})
        response["parsed_response"]["claims"][0]["text"] = "Children from 4 years."
        kinds = {flag["kind"] for flag in audit_one(prompt, response)["flags"]}
        self.assertEqual(kinds, {"claim_numeric_unit_absent_from_citations"})

    def test_passage_ids_and_numeric_units_are_extracted(self):
        blocks = passage_blocks("Question\n\n[P0001]\nCrCl 35 mL/min.\n\n[P0002]\nDose 10 mg.")
        self.assertEqual(set(blocks), {"P0001", "P0002"})
        self.assertEqual(numeric_units(blocks["P0001"]), {("35", "ml/min")})


if __name__ == "__main__":
    unittest.main()
