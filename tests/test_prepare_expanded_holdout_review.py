import unittest
from pathlib import Path

from labelqa.prepare_expanded_holdout_review import expansion_candidates
from labelqa.validate_holdout import validate_candidate_pairing


SOURCE = Path("data/snapshots/version_expansion_change_sources.jsonl")
PAIR = {"set_id": "11111111-2222-3333-4444-555555555555", "versions": [
    {"spl_version": 4, "published_date": "a", "xml_sha256": "1" * 64, "source_url": "u4"},
    {"spl_version": 5, "published_date": "b", "xml_sha256": "2" * 64, "source_url": "u5"},
]}


class ExpandedHoldoutReviewTests(unittest.TestCase):
    def test_same_question_is_applied_to_both_versions(self):
        rows = expansion_candidates({SOURCE: [PAIR]}, {PAIR["set_id"]: "What changed?"})
        self.assertEqual([row["qid"] for row in rows], ["vx-11111111-v4", "vx-11111111-v5"])
        self.assertEqual({row["question"] for row in rows}, {"What changed?"})
        self.assertEqual({row["source_snapshot"] for row in rows}, {SOURCE.as_posix()})
        validate_candidate_pairing(rows, {PAIR["set_id"]: SOURCE})

    def test_questions_must_cover_exactly_the_frozen_pairs(self):
        with self.assertRaises(ValueError):
            expansion_candidates({SOURCE: [PAIR]}, {})
        with self.assertRaises(ValueError):
            expansion_candidates({SOURCE: [PAIR]}, {PAIR["set_id"]: "What changed?",
                                                    "extra": "Another?"})

    def test_incomplete_question_draft_is_rejected(self):
        with self.assertRaises(ValueError):
            expansion_candidates({SOURCE: [PAIR]}, {PAIR["set_id"]: "TODO"})


if __name__ == "__main__":
    unittest.main()
