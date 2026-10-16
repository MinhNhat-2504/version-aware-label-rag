import json
import shutil
import unittest
from pathlib import Path

from labelqa.build_expansion_cohort import freeze_expansion, pre_exposure_checks
from labelqa.export_reviewer_interface import ROOT


def entry(index, version=3):
    return {"setid": f"label-{index}", "title": f"DRUG {index}", "spl_version": version,
            "published_date": "Oct 1, 2026"}


def history(set_id):
    if set_id == "label-gap":
        return [{"spl_version": 5, "published_date": "b"}, {"spl_version": 3, "published_date": "a"}]
    return [{"spl_version": 2, "published_date": "a"}, {"spl_version": 3, "published_date": "b"}]


def version(set_id, number):
    # Labels with an even suffix change their warnings text in the newer version.
    changed = set_id.split("-")[-1].isdigit() and int(set_id.split("-")[-1]) % 2 == 0
    text = "Old warning." if number == 2 or not changed else "New warning."
    return {"spl_version": number, "xml_sha256": f"{number:064x}",
            "sections": [{"section_index": 0, "code": "34071-1",
                          "title": "WARNINGS AND PRECAUTIONS", "text": text}]}


class ExpansionCohortTests(unittest.TestCase):
    def test_consecutive_arm_ignores_content_then_change_arm_screens_later_labels(self):
        entries = [entry(i) for i in range(1, 11)]
        consecutive, changed, screening = freeze_expansion(
            entries, excluded=set(), history=history, version=version,
            n_consecutive=3, n_changed=2, max_change_screen=5, now=lambda: "t")
        self.assertEqual([row["set_id"] for row in consecutive], ["label-1", "label-2", "label-3"])
        self.assertEqual([row["set_id"] for row in changed], ["label-4", "label-6"])
        self.assertEqual([row["decision"] for row in screening if row["arm"] == "change_screen"],
                         ["included_text_change", "no_clinical_section_text_change",
                          "included_text_change"])
        self.assertTrue(all(len(row["versions"]) == 2 for row in consecutive + changed))

    def test_exclusions_old_labels_and_broken_histories_are_not_sampled(self):
        entries = [entry(1), entry(2, version=1), {**entry(3), "setid": "label-gap"}, entry(4), entry(6)]
        consecutive, changed, screening = freeze_expansion(
            entries, excluded={"label-1"}, history=history, version=version,
            n_consecutive=1, n_changed=1, max_change_screen=3, now=lambda: "t")
        self.assertEqual([row["set_id"] for row in consecutive], ["label-4"])
        self.assertEqual([row["set_id"] for row in changed], ["label-6"])
        self.assertEqual(screening[0]["decision"], "excluded_download_or_history")
        self.assertNotIn("label-1", {row["set_id"] for row in screening})

    def test_change_screen_limit_stops_without_replacing_pairs(self):
        entries = [entry(i) for i in (1, 3, 5, 7, 9, 2)]
        _, changed, screening = freeze_expansion(
            entries, excluded=set(), history=history, version=version,
            n_consecutive=1, n_changed=1, max_change_screen=3, now=lambda: "t")
        self.assertEqual(changed, [])
        self.assertEqual(sum(row["arm"] == "change_screen" for row in screening), 3)


class PreExposureCheckTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "results" / ".test-expansion-fixture"
        shutil.rmtree(self.root, ignore_errors=True)
        (self.root / "results").mkdir(parents=True)
        (self.root / "data/review").mkdir(parents=True)
        (self.root / "results/holdout_source_preflight.json").write_text(json.dumps(
            {"independent_human_adjudications": 0, "holdout_model_evaluations": 0}), encoding="utf-8")
        blank = {"qid": "q", "reviewer_id": "", "item_decision": "", "independent_answer": "",
                 "required_claims": [], "acceptable_evidence_spans_per_claim": []}
        for reader in ("a", "b"):
            (self.root / f"data/review/version_holdout_reader_{reader}_blank.jsonl").write_text(
                json.dumps(blank) + "\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_blank_state_passes(self):
        self.assertTrue(pre_exposure_checks(self.root)["existing_blank_packets_unfilled"])

    def test_any_holdout_response_blocks_expansion(self):
        (self.root / "results/holdout_bm25_responses.jsonl").write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            pre_exposure_checks(self.root)

    def test_filled_reviewer_packet_blocks_expansion(self):
        path = self.root / "data/review/version_holdout_reader_b_blank.jsonl"
        path.write_text(json.dumps({"qid": "q", "reviewer_id": "reader-b"}) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            pre_exposure_checks(self.root)


if __name__ == "__main__":
    unittest.main()
