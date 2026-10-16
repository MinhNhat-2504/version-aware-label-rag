from pathlib import Path
from hashlib import sha256
import json
import unittest
from unittest.mock import patch
from uuid import uuid4

from labelqa.prepare_holdout_eval import load_manual_gold, passage_groups, prepare
from labelqa.spl_tables import StructuredPassage
from labelqa.validate_holdout import validate_candidate_pairing


class HoldoutEvaluationGateTests(unittest.TestCase):
    def test_two_reader_consensus_gold_passes_loader(self):
        stem = Path("results") / (".test_consensus_gold_" + uuid4().hex)
        gold_path = stem.with_suffix(".jsonl")
        manifest_path = gold_path.with_suffix(".manifest.json")
        sources = [stem.with_name(stem.name + f"_source_{index}.json") for index in range(4)]
        stem.parent.mkdir(parents=True, exist_ok=True)
        for path in [gold_path, manifest_path, *sources]:
            self.addCleanup(lambda item=path: item.unlink(missing_ok=True))
        for index, path in enumerate(sources):
            path.write_text(str(index), encoding="utf-8")
        pair = "11111111-1111-1111-1111-111111111111"
        rows = [{"qid": f"q-v{version}", "set_id": pair, "spl_version": version,
                 "source_snapshot": "version_holdout_sources.jsonl", "answerable": False,
                 "reference_answer": "Not established.", "claims": [],
                 "resolution_mode": "two_reader_consensus",
                 "resolver_ids": ["reader-a", "reader-b"], "adjudicator_id": "",
                 "adjudication_notes": "Both readers signed consensus."}
                for version in (1, 2)]
        gold_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = {"status": "two_reader_consensus_compiled_no_automatic_agreement_to_gold",
                    "candidate_items": 2, "accepted": 2, "excluded": 0, "exclusions": [],
                    "reader_ids": ["reader-a", "reader-b"],
                    "resolution_mode": "two_reader_consensus",
                    "resolver_ids": ["reader-a", "reader-b"],
                    "source_sha256": {str(path): sha256(path.read_bytes()).hexdigest()
                                      for path in sources},
                    "gold_sha256": sha256(gold_path.read_bytes()).hexdigest()}
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded, receipt = load_manual_gold(gold_path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(receipt["resolution_mode"], "two_reader_consensus")

    def test_version_candidates_share_question_and_source_pair(self):
        source = Path("frozen_synthetic_source.jsonl")
        pair = "11111111-1111-1111-1111-111111111111"
        rows = [{"qid": f"q-v{version}", "pair_id": pair, "set_id": pair,
                 "spl_version": version, "source_snapshot": str(source),
                 "question": "Same question?"} for version in (1, 2)]
        validate_candidate_pairing(rows, {pair: source})
        rows[1]["question"] = "A changed question?"
        with self.assertRaisesRegex(ValueError, "Question wording differs"):
            validate_candidate_pairing(rows, {pair: source})
        rows[1]["question"] = rows[0]["question"]
        rows[1]["source_snapshot"] = "other_source.jsonl"
        with self.assertRaisesRegex(ValueError, "pair/source identity mismatch"):
            validate_candidate_pairing(rows, {pair: source})

    def test_revised_packet_can_pair_more_than_one_question_per_version(self):
        source = Path("frozen_synthetic_source.jsonl")
        pair = "11111111-1111-1111-1111-111111111111"
        rows = [{"qid": f"q-{question}-{version}", "pair_id": pair, "set_id": pair,
                 "spl_version": version, "source_snapshot": str(source),
                 "question": question} for question in ("Question one?", "Question two?")
                for version in (1, 2)]
        validate_candidate_pairing(rows, {pair: source})
        rows[-1]["question"] = "Unpaired wording?"
        with self.assertRaisesRegex(ValueError, "Question wording differs across versions"):
            validate_candidate_pairing(rows, {pair: source})

    def test_candidate_file_cannot_be_used_as_manual_gold(self):
        with self.assertRaisesRegex(ValueError, "Manual adjudicated gold"):
            load_manual_gold(Path("data/holdout/version_question_candidates.jsonl"))

    def test_missing_gold_fails_before_source_or_model_access(self):
        missing = Path("data/holdout/__missing_manual_gold_for_test__.jsonl")
        self.assertFalse(missing.exists())
        with self.assertRaisesRegex(ValueError, "Manual adjudicated gold"):
            load_manual_gold(missing)

    def test_exact_quote_and_table_locator_map_to_atomic_passages(self):
        passages = [
            StructuredPassage(0, 7, "Interactions", "prose", "Avoid use in severe renal disease."),
            StructuredPassage(1, 7, "Interactions", "table_row",
                              "Drug: Alpha | Clinical comment: Stop Alpha during treatment.", 0, 2),
            StructuredPassage(2, 7, "Interactions", "table_row",
                              "Drug: Beta | Clinical comment: Stop Beta during treatment.", 0, 3),
        ]
        claims = [
            {"acceptable_evidence_spans": [{"section_index": 7, "source_kind": "prose",
                                              "quote": "severe renal disease"}]},
            {"acceptable_evidence_spans": [{"section_index": 7, "source_kind": "table_row",
                                              "table_index": 0, "row_index": 2,
                                              "quote": "Stop Alpha during treatment"}]},
        ]
        self.assertEqual(passage_groups(claims, passages, "q1"), [[0], [1]])
        claims[1]["acceptable_evidence_spans"][0]["row_index"] = 3
        with self.assertRaisesRegex(ValueError, "No parsed passage"):
            passage_groups(claims, passages, "q1")

    def test_synthetic_adjudicated_row_prepares_version_scoped_prompt(self):
        xml = b"synthetic pinned XML"
        set_id = "11111111-1111-1111-1111-111111111111"
        section = {"section_index": 7, "title": "Interactions", "code": "x",
                   "text": "Stop Alpha during treatment."}
        passage = StructuredPassage(0, 7, "Interactions", "table_row",
                                    "Drug: Alpha | Stop Alpha during treatment.", 0, 2)
        gold = {"qid": "q1", "set_id": set_id, "spl_version": 2,
                "source_snapshot": "synthetic-source.jsonl", "xml_sha256": sha256(xml).hexdigest(),
                "answerable": True, "question": "What is the Alpha instruction?",
                "reference_answer": "manual gold answer marker",
                "claims": [{"text": "Stop Alpha", "acceptable_evidence_spans": [
                    {"section_index": 7, "source_kind": "table_row", "table_index": 0,
                     "row_index": 2, "quote": "Stop Alpha during treatment."}]}]}
        source = {"set_id": set_id, "title": "Synthetic Drug", "versions": [
            {"spl_version": 2, "xml_sha256": sha256(xml).hexdigest(), "sections": [section]}]}
        with (patch("labelqa.prepare_holdout_eval.read_jsonl", return_value=iter([source])),
              patch.object(Path, "read_bytes", return_value=xml),
              patch("labelqa.prepare_holdout_eval.parse_sections", return_value=[section]),
              patch("labelqa.prepare_holdout_eval.parse_structured_passages", return_value=[passage])):
            prompt = prepare([gold], Path("unused"), "bm25")[0]
        self.assertEqual(prompt["gold_claims"], [[0]])
        self.assertEqual(prompt["retrieved_indices"], [0])
        self.assertEqual(prompt["passage_map"][0]["row_index"], 2)
        self.assertIn("Target SPL version: 2", prompt["user_prompt"])
        self.assertNotIn("manual gold answer marker", prompt["user_prompt"])


if __name__ == "__main__":
    unittest.main()
