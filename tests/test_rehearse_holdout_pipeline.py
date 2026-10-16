import unittest

from labelqa.answer_review import validate_review
from labelqa.rehearse_holdout_pipeline import synthetic_answer_review


def packet_row(answer, abstain=False):
    claims = [] if abstain else [{"text": "A claim.", "citations": ["P0001"]}]
    return {"review_id": "r" * 16, "qid": "q1", "visible_evidence": "[P0001]\nSection text",
            "model_answer": {"answer": answer, "abstain": abstain, "claims": claims},
            "raw_model_output": "{}", "generation_error": False,
            "reviewer_id": "", "review_status": "", "answer_complete": None,
            "all_claims_supported": None, "citations_at_correct_location": None,
            "correct_version": None, "refusal_appropriate": None,
            "unsafe_unsupported_claim": None, "claim_judgments": [], "review_notes": ""}


class RehearsalReviewTests(unittest.TestCase):
    def test_placeholder_review_is_valid_and_never_claims_support(self):
        blank = packet_row("An answer.")
        review = synthetic_answer_review(blank, "reader", {"q1": True})
        validate_review(blank, review)
        self.assertFalse(review["all_claims_supported"])
        self.assertFalse(review["answer_complete"])
        self.assertEqual({j["support"] for j in review["claim_judgments"]}, {"uncertain"})

    def test_refusal_review_uses_gold_answerability(self):
        blank = packet_row("Not stated in this version.", abstain=True)
        review = synthetic_answer_review(blank, "reader", {"q1": False})
        validate_review(blank, review)
        self.assertTrue(review["refusal_appropriate"])

    def test_invalid_output_is_marked_without_judgments(self):
        blank = {**packet_row("x"), "model_answer": None}
        review = synthetic_answer_review(blank, "reader", {"q1": True})
        validate_review(blank, review)
        self.assertEqual(review["review_status"], "invalid_output")


if __name__ == "__main__":
    unittest.main()
