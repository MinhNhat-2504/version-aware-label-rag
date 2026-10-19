"""The AI silver-standard rule (Amendment 6) and AI reviewer detection."""

import unittest

from labelqa.compile_silver_standard import silver_rows
from labelqa.review_resolution import SILVER_MODE, is_llm, review_design


def candidate(qid):
    return {"qid": qid, "pair_id": "p", "set_id": "p", "spl_version": 1, "xml_sha256": "x",
            "source_snapshot": "version_holdout_sources.jsonl", "question": "Q?"}


def review(decision="accept", answerable=True, change="stable", claim="Claim."):
    span = {"section_index": 0, "source_kind": "prose", "quote": "q", "table_index": None, "row_index": None}
    return {"item_decision": decision, "answerable_from_version": answerable if decision == "accept" else None,
            "version_change_judgment": change if decision == "accept" else "",
            "independent_answer": "Answer." if decision == "accept" else "",
            "required_claims": [claim] if answerable and decision == "accept" else [],
            "acceptable_section_indices_per_claim": [[0]] if answerable and decision == "accept" else [],
            "acceptable_evidence_spans_per_claim": [[span]] if answerable and decision == "accept" else []}


class SilverRuleTests(unittest.TestCase):
    def test_rule_takes_codex_claims_and_excludes_disagreements(self):
        qids = ["agree", "change", "rewrite", "answerability"]
        llm = {"agree": review(claim="LLM claim."), "change": review(change="text_only"),
               "rewrite": review(decision="rewrite"), "answerability": review(answerable=True)}
        codex = {"agree": review(claim="Codex claim."), "change": review(change="material"),
                 "rewrite": review(decision="reject"), "answerability": review(answerable=False)}
        accepted, excluded = silver_rows([candidate(q) for q in qids], llm, codex)
        self.assertEqual([row["qid"] for row in accepted], ["agree", "change"])
        self.assertEqual(accepted[0]["claims"][0]["text"], "Codex claim.")
        self.assertEqual(accepted[1]["version_change_judgment"], "uncertain")
        self.assertTrue(all(row["resolution_mode"] == SILVER_MODE and row["resolver_ids"] == [] for row in accepted))
        reasons = {row["qid"]: row["reason"] for row in excluded}
        self.assertIn("not accepted by both", reasons["rewrite"])
        self.assertIn("answerability", reasons["answerability"])

    def test_ai_ids_never_pass_as_human(self):
        for ai in ("llm_annotator_claude", "codex_ai_expert_simulation", "ai_reader", "reader_ai_2"):
            self.assertTrue(is_llm(ai), ai)
        for human in ("expert_01", "reader_a_01", "rehearsal-expert", "claire_01", "maida"):
            self.assertFalse(is_llm(human), human)
        design = review_design({"reader_ids": ["llm_a", "codex_b"], "resolution_mode": SILVER_MODE,
                                "resolver_ids": []})
        self.assertEqual(design["agreement_kind"], "ai_ai_agreement_not_reliability")
        self.assertFalse(design["human_review"])


if __name__ == "__main__":
    unittest.main()
