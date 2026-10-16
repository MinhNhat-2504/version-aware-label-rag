import json
import unittest

from labelqa.export_reviewer_interface import ROOT, build_payload, write_bundle


class ReviewerInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / "results" / ".test-reviewer-fixture"
        packet = cls.root / "data/review/version_holdout_reader_a_blank.jsonl"
        snapshot = cls.root / "data/snapshots/version_holdout_sources.jsonl"
        packet.parent.mkdir(parents=True, exist_ok=True)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        packet_rows = []
        source_rows = []
        for pair_index in range(12):
            set_id = f"00000000-0000-0000-0000-{pair_index:012d}"
            versions = []
            for offset in (1, 2):
                qid = f"fixture-{pair_index}-v{offset}"
                digest = f"{pair_index * 2 + offset:064x}"
                versions.append({
                    "published_date": f"Oct {offset}, 2026", "spl_version": offset,
                    "source_url": f"https://example.invalid/{set_id}/v{offset}",
                    "xml_sha256": digest,
                    "zip_sha256": f"{pair_index * 2 + offset + 100:064x}",
                    "sections": [{"section_index": 1, "title": "INDICATIONS",
                                  "code": "34067-9", "text": f"Fixture evidence {qid}."}],
                })
                packet_rows.append({
                    "qid": qid, "pair_id": set_id, "set_id": set_id,
                    "spl_version": offset, "published_date": f"Oct {offset}, 2026",
                    "xml_sha256": digest,
                    "source_url": f"https://example.invalid/{set_id}/v{offset}",
                    "source_snapshot": r"data\snapshots\version_holdout_sources.jsonl",
                    "question": f"Fixture question {pair_index}?",
                    "question_origin": "synthetic_test_fixture",
                    "annotation_status": "candidate_pending_two_independent_readers",
                    "reviewer_id": "", "item_decision": "", "independent_answer": "",
                    "answerable_from_version": None, "required_claims": [],
                    "acceptable_section_indices_per_claim": [],
                    "version_change_judgment": "", "clinical_significance": "", "notes": "",
                    "acceptable_evidence_spans_per_claim": [],
                })
            source_rows.append({
                "set_id": set_id, "title": f"FIXTURE DRUG {pair_index}",
                "listed_published_date": "Oct 2, 2026", "retrieved_utc": "2026-10-16T00:00:00Z",
                "versions": versions, "annotation_status": "synthetic_test_fixture",
            })
        packet.write_text("".join(json.dumps(row) + "\n" for row in packet_rows), encoding="utf-8")
        snapshot.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")

    def test_payload_contains_sources_without_model_outputs(self):
        payload = build_payload(self.root)
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertFalse(payload["model_outputs_included"])
        self.assertEqual(len(payload["pairs"]), 12)
        self.assertEqual(len(payload["packet_rows"]), 24)
        self.assertTrue(all(len(pair["items"]) == 2 for pair in payload["pairs"]))
        self.assertTrue(all(item["sections"] for pair in payload["pairs"] for item in pair["items"]))
        forbidden = {"model_answer", "retrieved_passages", "prediction", "gold_answer"}
        self.assertFalse(forbidden & set(json.dumps(payload).split('"')))

    def test_bundle_is_a_browser_script_with_roundtrippable_json(self):
        payload = build_payload(self.root)
        output = self.root / "review-data.js"
        try:
            write_bundle(payload, output)
            text = output.read_text(encoding="utf-8")
            prefix = "window.LABELQA_REVIEW_DATA="
            self.assertTrue(text.startswith(prefix))
            decoded = json.loads(text[len(prefix):-2])
            self.assertEqual(decoded["inputs_sha256"], payload["inputs_sha256"])
        finally:
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
