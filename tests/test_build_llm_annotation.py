import shutil
import unittest
from hashlib import sha256

from labelqa.build_llm_annotation import build_rows
from labelqa.export_reviewer_interface import ROOT
from labelqa.synthetic_demo import SYNTHETIC_XML


SET_ID = "00000000-0000-0000-0000-000000000101"


def candidate():
    return {"qid": "q-v2", "pair_id": SET_ID, "set_id": SET_ID, "spl_version": 2,
            "xml_sha256": sha256(SYNTHETIC_XML).hexdigest(), "question": "What schedule?"}


def item(evidence, answerable=True):
    return {"decision": "accept", "answerable": answerable, "answer": "Once daily.",
            "claims": [{"text": "Adults take one unit once daily.", "evidence": evidence}] if answerable else [],
            "version_change": "stable"}


class LlmAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.xml_dir = ROOT / "results" / ".test-llm-annotation"
        shutil.rmtree(self.xml_dir, ignore_errors=True)
        self.xml_dir.mkdir(parents=True)
        (self.xml_dir / f"{SET_ID}_v2.xml").write_bytes(SYNTHETIC_XML)

    def tearDown(self):
        shutil.rmtree(self.xml_dir, ignore_errors=True)

    def test_table_quote_resolves_to_its_row_under_llm_id(self):
        spec = {"annotator_id": "llm_annotator_test", "items": {"q-v2": item(
            [{"section_index": 1, "quote": "One synthetic unit once daily.", "kind": "table_row"}])}}
        row = build_rows(spec, [candidate()], self.xml_dir)[0]
        self.assertEqual(row["reviewer_id"], "llm_annotator_test")
        span = row["acceptable_evidence_spans_per_claim"][0][0]
        self.assertEqual((span["table_index"], span["row_index"]), (0, 1))
        self.assertEqual(row["acceptable_section_indices_per_claim"], [[1]])

    def test_human_id_and_unplaceable_quotes_are_rejected(self):
        good = item([{"section_index": 0, "quote": "indicated to treat Condition Alpha"}])
        with self.assertRaisesRegex(ValueError, "llm_"):
            build_rows({"annotator_id": "reader_a_01", "items": {"q-v2": good}}, [candidate()], self.xml_dir)
        bad = item([{"section_index": 0, "quote": "in adults. Store the synthetic package"}])
        with self.assertRaisesRegex(ValueError, "not inside one parsed window"):
            build_rows({"annotator_id": "llm_x", "items": {"q-v2": bad}}, [candidate()], self.xml_dir)

    def test_annotation_must_cover_every_candidate(self):
        with self.assertRaisesRegex(ValueError, "exactly the candidate items"):
            build_rows({"annotator_id": "llm_x", "items": {}}, [candidate()], self.xml_dir)


if __name__ == "__main__":
    unittest.main()
