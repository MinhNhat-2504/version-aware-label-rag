"""The disclosed-LLM plus human-expert source review path, on a synthetic label."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import unittest
import uuid

from labelqa.check_holdout_reviews import blind_receipt, check
from labelqa.compile_holdout_gold import compile_gold
from labelqa.export_reviewer_interface import ROOT
from labelqa.recent import parse_sections
from labelqa.review_agreement import cohen_kappa
from labelqa.spl_tables import parse_structured_passages
from labelqa.synthetic_demo import SYNTHETIC_XML


SET_ID = "00000000-0000-0000-0000-000000000202"
LLM, EXPERT = "llm_annotator_test", "expert_01"
PROSE = "SYNTH-101 is indicated to treat Condition Alpha in adults."
ROW = "One synthetic unit once daily."


def span(section, kind, quote, table=None, row=None):
    return {"section_index": section, "source_kind": kind, "quote": quote,
            "table_index": table, "row_index": row}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class LlmExpertSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "results" / f".test-llm-expert-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        self.xml_dir = self.root / "xml"
        self.xml_dir.mkdir()
        (self.xml_dir / f"{SET_ID}_v2.xml").write_bytes(SYNTHETIC_XML)
        sections = parse_sections(SYNTHETIC_XML)
        self.sections = {s["section_index"]: s["text"] for s in sections}
        snapshot = self.root / "sources.jsonl"
        write_jsonl(snapshot, [{"set_id": SET_ID, "title": "SYNTH-101", "versions": [
            {"spl_version": 2, "xml_sha256": sha256(SYNTHETIC_XML).hexdigest(), "sections": sections}]}])
        self.candidate = {"qid": "q-v2", "pair_id": SET_ID, "set_id": SET_ID, "spl_version": 2,
                          "xml_sha256": sha256(SYNTHETIC_XML).hexdigest(),
                          "source_snapshot": snapshot.as_posix(), "question": "What is it for?"}
        self.candidates = self.root / "candidates.jsonl"
        write_jsonl(self.candidates, [self.candidate])
        review = {**self.candidate, "item_decision": "accept", "independent_answer": PROSE,
                  "answerable_from_version": True, "required_claims": [PROSE, ROW],
                  "acceptable_section_indices_per_claim": [[0], [1]],
                  "acceptable_evidence_spans_per_claim": [[span(0, "prose", PROSE)],
                                                          [span(1, "table_row", ROW, 0, 1)]],
                  "version_change_judgment": "stable"}
        self.llm = self.root / "llm.jsonl"
        self.expert = self.root / "expert.jsonl"
        write_jsonl(self.llm, [{**review, "reviewer_id": LLM}])
        write_jsonl(self.expert, [{**review, "reviewer_id": EXPERT, "independent_answer": "Condition Alpha."}])
        self.provenance = self.root / "llm.provenance.json"
        self.provenance.write_text(json.dumps({
            "status": "disclosed_llm_annotation_not_human_review_not_gold", "annotator_id": LLM,
            "annotation_sha256": sha256(self.llm.read_bytes()).hexdigest(),
            "candidates_sha256": sha256(self.candidates.read_bytes()).hexdigest()}), encoding="utf-8")
        self.receipt = self.root / "expert.receipt.json"
        report = check(self.candidates, self.expert, None, self.xml_dir)
        self.receipt.write_text(json.dumps(blind_receipt(report, self.expert, self.candidates)),
                                encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def queue(self):
        return check(self.candidates, self.llm, self.expert, self.xml_dir, self.receipt, self.provenance)

    def test_queue_records_receipts_and_expert_llm_agreement(self):
        queue = self.queue()
        self.assertEqual(queue["reader_ids"], [LLM, EXPERT])
        self.assertEqual(queue["agreement"]["kind"], "expert_llm_agreement_not_inter_human_reliability")
        self.assertEqual(queue["agreement"]["item_decision"]["agree"], 1)
        self.assertEqual(queue["agreement"]["identical_answer_text_when_both_accept"], 0)
        self.assertEqual(queue["agreement"]["distinct_quotes"]["identical"], 2)
        self.assertEqual(queue["blind_receipts"]["human_reader"]["reader_sha256"],
                         sha256(self.expert.read_bytes()).hexdigest())

    def test_llm_queue_needs_receipts_and_detects_later_edits(self):
        with self.assertRaisesRegex(ValueError, "blind receipt and the LLM provenance"):
            check(self.candidates, self.llm, self.expert, self.xml_dir)
        self.expert.write_text(self.expert.read_text(encoding="utf-8").replace("Condition Alpha.", "Edited."),
                               encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed after freezing"):
            self.queue()

    def test_llm_file_must_match_provenance(self):
        self.llm.write_text(self.llm.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance receipt"):
            self.queue()

    def test_blind_receipt_is_for_humans_only(self):
        report = check(self.candidates, self.llm, None, self.xml_dir)
        with self.assertRaisesRegex(ValueError, "provenance receipt"):
            blind_receipt(report, self.llm, self.candidates)

    def test_expert_resolution_compiles_and_final_locators_are_checked(self):
        queue = self.queue()
        signed = deepcopy(queue)
        row = signed["queue"][0]
        expert = deepcopy(row["reader_b"])
        row.update({"resolution_mode": "human_expert_resolution", "expert_resolver_id": EXPERT,
                    "final_decision": "accept", "final_answer": expert["independent_answer"],
                    "final_answerable_from_version": True, "final_claims": expert["required_claims"],
                    "final_acceptable_section_indices_per_claim": expert["acceptable_section_indices_per_claim"],
                    "final_acceptable_evidence_spans_per_claim": expert["acceptable_evidence_spans_per_claim"],
                    "final_version_change_judgment": "stable", "adjudication_notes": "Kept my answer."})
        passages = {"q-v2": parse_structured_passages(SYNTHETIC_XML)}
        gold, excluded = compile_gold(queue, signed, {"q-v2": self.candidate}, {"q-v2": self.sections}, passages)
        self.assertEqual((len(gold), excluded), (1, []))
        self.assertEqual(gold[0]["resolver_ids"], [EXPERT])
        # A final quote inside the section text but across two parsed passages fails now, not after gold.
        crossing = deepcopy(signed)
        crossing["queue"][0]["final_acceptable_evidence_spans_per_claim"][1] = [
            span(1, "table_row", "Schedule Adults", 0, 1)]
        with self.assertRaisesRegex(ValueError, "not inside one parsed"):
            compile_gold(queue, crossing, {"q-v2": self.candidate}, {"q-v2": self.sections}, passages)

    def test_cohen_kappa(self):
        self.assertEqual(cohen_kappa([("a", "a"), ("b", "b")]), 1.0)
        self.assertIsNone(cohen_kappa([("a", "a"), ("a", "a")]))
        self.assertIsNone(cohen_kappa([]))


if __name__ == "__main__":
    unittest.main()
