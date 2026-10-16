"""A synthetic source exercises the technical path without exposing holdout answers."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from labelqa.check_holdout_reviews import check
from labelqa.compile_holdout_gold import compile_gold
from labelqa.learned_retrieval import FEATURE_NAMES
from labelqa.prepare_blinded_answer_review import review_packet
from labelqa.prepare_holdout_eval import prepare
from labelqa.recent import parse_sections
from labelqa.score_qa import score_one, validate_provenance
from labelqa.summarize_holdout_retrieval import METHODS, summarize


class SyntheticHoldoutPipelineTests(unittest.TestCase):
    def test_manual_gold_to_prompts_scores_and_blinded_packet(self):
        xml = (b'<ClinicalDocument xmlns="urn:hl7-org:v3"><section><title>Indications</title>'
               b'<text>Drug Alpha prevents events.<table><thead><tr><th>Drug</th>'
               b'<th>Clinical comment</th></tr></thead><tbody><tr><td>Drug Beta</td>'
               b'<td>Stop during treatment.</td></tr></tbody></table></text>'
               b'</section></ClinicalDocument>')
        set_id = "11111111-1111-1111-1111-111111111111"
        source_path = "version_holdout_sources.jsonl"
        question = "What does Drug Alpha do, and what is the instruction for Drug Beta?"
        prose = "Drug Alpha prevents events."
        row_claim = "Stop Drug Beta during treatment."
        section = parse_sections(xml)[0]
        candidate = {"qid": "q-synthetic", "pair_id": set_id, "set_id": set_id,
                     "spl_version": 2, "xml_sha256": sha256(xml).hexdigest(),
                     "source_snapshot": source_path, "question": question}
        source = {"set_id": set_id, "title": "Drug Alpha", "versions": [
            {"spl_version": 2, "xml_sha256": sha256(xml).hexdigest(), "sections": [section]}]}
        span = {"section_index": 0, "source_kind": "prose", "quote": prose}
        row_span = {"section_index": 0, "source_kind": "table_row", "table_index": 0,
                    "row_index": 1, "quote": "Stop during treatment."}
        review = {**candidate, "item_decision": "accept", "independent_answer": prose,
                  "answerable_from_version": True, "required_claims": [prose, row_claim],
                  "acceptable_section_indices_per_claim": [[0], [0]],
                  "acceptable_evidence_spans_per_claim": [[span], [row_span]],
                  "version_change_judgment": "stable"}
        paths = [Path("candidate"), Path("reader-a"), Path("reader-b")]
        indexed = {paths[0]: {candidate["qid"]: candidate},
                   paths[1]: {candidate["qid"]: {**review, "reviewer_id": "reader-a"}},
                   paths[2]: {candidate["qid"]: {**review, "reviewer_id": "reader-b"}}}
        with (patch("labelqa.check_holdout_reviews.index_reviews", side_effect=lambda path: indexed[path]),
              patch("labelqa.check_holdout_reviews.read_jsonl", return_value=iter([source]))):
            queue = check(*paths)
        edited = deepcopy(queue)
        edited["queue"][0].update({
            "resolution_mode": "two_reader_consensus",
            "consensus_reviewer_ids": ["reader-a", "reader-b"],
            "adjudicator_id": "", "final_decision": "accept",
            "final_answerable_from_version": True, "final_answer": prose,
            "final_claims": [prose, row_claim],
            "final_acceptable_section_indices_per_claim": [[0], [0]],
            "final_acceptable_evidence_spans_per_claim": [[span], [row_span]],
            "final_version_change_judgment": "stable",
            "adjudication_notes": "Both synthetic readers signed this integration fixture.",
        })
        gold, excluded = compile_gold(queue, edited, {candidate["qid"]: candidate},
                                      {candidate["qid"]: {0: section["text"]}})
        self.assertEqual(excluded, [])
        with (patch("labelqa.prepare_holdout_eval.read_jsonl", side_effect=lambda _: iter([source])),
              patch.object(Path, "read_bytes", return_value=xml),
              patch("labelqa.prepare_holdout_eval.model_weights",
                    return_value=([0.0] * len(FEATURE_NAMES), "synthetic-selector-hash"))):
            prompts = {method: {candidate["qid"]: prepare(gold, Path("unused"), method)[0]}
                       for method in METHODS}
        responses = {}
        for method in METHODS:
            prompt = prompts[method][candidate["qid"]]
            self.assertEqual(prompt["gold_claims"], [[0], [1]])
            response = {"qid": candidate["qid"], "task": prompt["task"],
                        "prompt_sha256": sha256((prompt["system_prompt"] +
                                                 prompt["user_prompt"]).encode()).hexdigest(),
                        "model_digest": "synthetic-model", "decoding_options": {"temperature": 0},
                        "parsed_response": {"answer": prose, "abstain": False,
                                            "claims": [{"text": prose, "citations": ["P0000"]},
                                                       {"text": row_claim, "citations": ["P0001"]}]},
                        "raw_response": "synthetic output"}
            responses[method] = {candidate["qid"]: response}
            score = score_one(prompt, response)
            self.assertTrue(score["retrieval_complete_evidence"])
            self.assertTrue(score["valid_schema"])
            validate_provenance(prompts[method], responses[method])
        summary = summarize(gold, prompts)
        self.assertEqual(summary["answerable_items"], 1)
        self.assertEqual(summary["learned_vs_bm25"]["paired_cells"]["both"], 1)
        ids = iter(f"blind-synthetic-{i}" for i in range(3))
        packet, private_key = review_packet(gold, prompts, responses, id_factory=lambda: next(ids))
        self.assertEqual(len(packet), 3)
        self.assertEqual({row["method"] for row in private_key}, set(METHODS))
        self.assertTrue(all("method" not in row and "reference_answer" not in row for row in packet))


if __name__ == "__main__":
    unittest.main()
