"""The answer-review interface bundle and its browser rules agree with the Python checker."""

import json
from pathlib import Path
import shutil
import subprocess
import unittest
import uuid

from labelqa.answer_review import check, packet_index, validate_review
from labelqa.export_answer_review_interface import build_payload
from labelqa.export_reviewer_interface import ROOT
from labelqa.prepare_blinded_answer_review import review_packet
from labelqa.summarize_holdout_retrieval import METHODS


LOGIC = ROOT / "answer_review_ui" / "dist" / "logic.js"
NODE_HARNESS = r"""
const L = require(process.argv[1]);
const payload = JSON.parse(require("fs").readFileSync(process.argv[2], "utf8"));
const out = { reviews: [], blankErrors: [], llmErrors: 0, evidence: L.splitEvidence(payload.items[0].row.visible_evidence) };
for (const item of payload.items) {
  const kind = L.itemKind(item);
  const state = L.blankState(item);
  out.blankErrors.push(L.itemErrors(item, state, "expert_01").length);
  state.review_notes = "Checked against the full label.";
  if (kind !== "invalid") { state.correct_version = true; state.unsafe_unsupported_claim = false; }
  if (kind === "refusal") state.refusal_appropriate = false;
  if (kind === "answer") {
    state.answer_complete = true;
    state.claims.forEach((claim, index) => {
      claim.support = index === 0 ? "unsupported" : "supported";
      claim.citation_location_valid = true;
      claim.notes = index === 0 ? "Relation not stated in the cited row." : "";
    });
  }
  if (L.itemErrors(item, state, "expert_01").length) throw new Error("valid state rejected: " + kind);
  if (L.itemErrors(item, state, "llm_x").length) out.llmErrors += 1;
  const review = L.buildReview(item, state, "expert_01");
  const restored = L.stateFromReview(item, review);
  if (JSON.stringify(L.buildReview(item, restored, "expert_01")) !== JSON.stringify(review)) throw new Error("round trip");
  out.reviews.push(review);
}
let refused = false;
try { L.checkImportedRows(payload.items, out.reviews.map(r => ({ ...r, reviewer_id: "llm_x" }))); } catch (e) { refused = true; }
out.llmImportRefused = refused;
L.checkImportedRows(payload.items, out.reviews);
process.stdout.write(JSON.stringify(out));
"""


class AnswerReviewInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "results" / f".test-answer-ui-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True)
        snapshot = self.root / "version_holdout_sources.jsonl"
        snapshot.write_text(json.dumps({"set_id": "pair-1", "title": "SYNTH-101", "versions": [
            {"spl_version": version, "xml_sha256": f"hash-{version}",
             "sections": [{"section_index": 0, "title": "INDICATIONS", "code": "34067-9",
                           "text": "Synthetic indication text."}]} for version in (1, 2)]}) + "\n",
            encoding="utf-8")
        gold = [{"qid": f"q{version}", "set_id": "pair-1", "spl_version": version,
                 "xml_sha256": f"hash-{version}", "source_snapshot": snapshot.as_posix(),
                 "question": "Is it indicated?", "answerable": version == 1} for version in (1, 2)]
        prompts, responses = {m: {} for m in METHODS}, {m: {} for m in METHODS}
        for method in METHODS:
            for row in gold:
                prompts[method][row["qid"]] = {
                    **{k: row[k] for k in ("qid", "set_id", "spl_version", "xml_sha256", "question")},
                    "user_prompt": "Question\nLabel passages:\n[P0000]\nSPL section 0: INDICATIONS (prose)\n"
                                   "Synthetic indication text.\n\n[P0003]\nSPL section 0: INDICATIONS (prose)\nMore."}
                parsed = ({"answer": "Not stated.", "abstain": True, "claims": []} if row["qid"] == "q2" else
                          {"answer": "Indicated.", "abstain": False,
                           "claims": [{"text": "It is indicated.", "citations": ["P0000"]},
                                      {"text": "Second claim.", "citations": ["P0003", "P0009"]}]})
                if method == "ontology_cover" and row["qid"] == "q1":
                    parsed = None
                responses[method][row["qid"]] = {"prompt_sha256": "h", "model_digest": "m",
                                                 "decoding_options": {"temperature": 0}, "parsed_response": parsed,
                                                 "raw_response": "raw", "error": None}
        serial = iter(f"blind-answer-{i:02d}" for i in range(6))
        self.packet_rows, _ = review_packet(gold, prompts, responses, id_factory=lambda: next(serial))
        self.packet = self.root / "packet.jsonl"
        self.packet.write_text("".join(json.dumps(r) + "\n" for r in self.packet_rows), encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_bundle_has_answers_sections_and_no_method_labels(self):
        payload = build_payload(self.packet, self.root)
        self.assertTrue(payload["model_outputs_included"])
        self.assertFalse(payload["method_labels_included"])
        self.assertEqual(len(payload["items"]), 6)
        self.assertEqual(len(payload["labels"]), 2)
        kinds = sorted("invalid" if not i["machine"]["valid_schema"] else
                       "refusal" if i["machine"]["abstain"] else "answer" for i in payload["items"])
        self.assertEqual(kinds, ["answer", "answer", "invalid", "refusal", "refusal", "refusal"])
        self.assertNotIn("method", json.dumps(payload["items"]))

    def test_packet_naming_a_method_is_refused(self):
        leaked = [{**row, "method": "bm25"} for row in self.packet_rows]
        self.packet.write_text("".join(json.dumps(r) + "\n" for r in leaked), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "method or gold"):
            build_payload(self.packet, self.root)

    @unittest.skipUnless(shutil.which("node"), "Node.js is optional")
    def test_browser_rules_produce_rows_the_python_checker_accepts(self):
        payload = build_payload(self.packet, self.root)
        bundle = self.root / "payload.json"
        bundle.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(["node", "-e", NODE_HARNESS, str(LOGIC), str(bundle)],
                                capture_output=True, text=True, encoding="utf-8", check=True)
        out = json.loads(result.stdout)
        self.assertTrue(all(count > 0 for count in out["blankErrors"]))
        self.assertEqual(out["llmErrors"], 6)
        self.assertTrue(out["llmImportRefused"])
        self.assertEqual([p["id"] for p in out["evidence"]], ["P0000", "P0003"])
        blank = packet_index(self.packet_rows)
        for review in out["reviews"]:
            validate_review(blank[review["review_id"]], review)
        first = {r["review_id"]: r for r in out["reviews"]}
        second = {rid: {**r, "reviewer_id": "llm_answer_reviewer"} for rid, r in first.items()}
        queue = check(self.packet_rows, first, second, ("expert_01", "llm_answer_reviewer"))
        self.assertEqual(queue["items"], 6)


if __name__ == "__main__":
    unittest.main()
