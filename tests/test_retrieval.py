from pathlib import Path
from hashlib import sha256
import unittest
from unittest.mock import patch

from labelqa.data import Label, Question, load_questions, split_label
from labelqa.evaluate import summarize
from labelqa.learned_retrieval import FEATURE_NAMES, feature_pool, retrieve as learned_retrieve, train
from labelqa.prompts import make_prompt
from labelqa.recent import parse_sections
from labelqa.score_qa import score_one, validate_provenance
from labelqa.spl_passages import split_sections
from labelqa.spl_tables import parse_structured_passages
from labelqa.retrieval import LabelIndex
from labelqa.verify_claims import contraindication_flag
from labelqa.version_gate import absent_from_label, target_entity


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = LabelIndex(
            Label(
                "label-one",
                "Example medicine",
                (
                    "1 INDICATIONS\nUsed to treat insomnia.",
                    "2 DOSAGE\nTake one 10 mg tablet each evening.",
                    "7 INTERACTIONS\nNaltrexone reduces the drug effect.",
                ),
            )
        )

    def test_relevant_passage_is_first(self):
        self.assertEqual(self.index.retrieve("How does it interact with naltrexone?", k=1), [2])

    def test_tfidf_and_lexical_fusion_respect_evidence_budget(self):
        for method in ("tfidf", "rrf_lexical"):
            ranking = self.index.retrieve("How does it interact with naltrexone?", method, k=2)
            self.assertEqual(ranking[0], 2)
            self.assertEqual(len(ranking), len(set(ranking)))
            self.assertEqual(len(ranking), 2)

    def test_learned_selector_uses_candidate_features_without_gold(self):
        features, pool = feature_pool(self.index, "How does it interact with naltrexone?")
        self.assertEqual(set(features), set(pool))
        weights = [0.0] * len(FEATURE_NAMES)
        weights[0] = 1.0
        for coverage in (False, True):
            ranking = learned_retrieve(self.index, "How does it interact with naltrexone?",
                                       weights, k=2, coverage=coverage)
            self.assertEqual(ranking[0], 2)
            self.assertEqual(len(ranking), 2)

    def test_learned_ablation_keeps_disabled_weights_zero(self):
        positive = (1.0,) * len(FEATURE_NAMES)
        negative = (0.0,) * len(FEATURE_NAMES)
        weights, metadata = train([({0: positive, 1: negative}, [0, 1], {0})],
                                  frozenset(("tfidf_relative", "tfidf_reciprocal_rank")))
        self.assertEqual(weights[2:4], [0.0, 0.0])
        self.assertEqual(len(metadata["disabled_features"]), 2)

    def test_both_evidence_passages_can_be_retrieved(self):
        for method in ("bm25", "section_cover", "clause_cover", "ontology_cover"):
            ranking = self.index.retrieve("What dosage and interaction with naltrexone?", method, k=2)
            self.assertEqual(set(ranking), {1, 2})

    def test_evidence_set_reserves_indication_and_ecg_sections(self):
        label = Label("label-two", "Example medicine", (
            "1 INDICATIONS AND USAGE\nUsed in adults after prior therapy.",
            "5.3 Electrocardiographic Changes\nMonitor ECG and electrolytes.",
            "6 ADVERSE REACTIONS\nMany adults experienced fatigue after therapy.",
        ))
        ranking = LabelIndex(label).retrieve(
            "Which adults is it indicated for, and what ECG precautions apply?",
            "evidence_set", k=2,
        )
        self.assertEqual(set(ranking), {0, 1})

    def test_multihop_all_gold_metric(self):
        question = Question("q1", "label-one", "multihop", "Question", (1, 2))
        result = summarize([(question, [2, 1, 0])])
        self.assertEqual(result["multihop"]["all_gold@5"], 1.0)
        self.assertEqual(result["multihop"]["recall@1"], 0.5)

    def test_label_split_is_stable(self):
        self.assertEqual(split_label("label-one"), split_label("label-one"))

    def test_gold_uses_passage_indices_not_section_labels(self):
        record = {
            "qid": "q1", "set_id": "label-one", "task": "multihop",
            "question": "Question", "citations": ["2.1", "7"],
            "context": [{"doc_chunk_index": 3}, {"doc_chunk_index": 8}],
        }
        with patch("labelqa.data.read_jsonl", return_value=iter([record])):
            self.assertEqual(load_questions(Path("unused"))[0].gold, (3, 8))

    def test_prompt_uses_exact_retrieved_passage_ids(self):
        prompt = make_prompt(self.index.label, "Dose?", [1, 2])
        self.assertIn("[P0001]", prompt["user_prompt"])
        self.assertIn("[P0002]", prompt["user_prompt"])
        self.assertNotIn("[P0000]", prompt["user_prompt"])

    def test_spl_sections_keep_codes_without_nested_text_duplication(self):
        xml = b'''<document xmlns="urn:hl7-org:v3"><component><section>
          <title>Warnings</title><code code="34071-1"/><text>Watch for <content>rash</content>.</text>
          <component><section><title>Renal</title><text>Reduce dose.</text></section></component>
        </section></component></document>'''
        sections = parse_sections(xml)
        self.assertEqual([section["title"] for section in sections], ["Warnings", "Renal"])
        self.assertEqual(sections[0]["code"], "34071-1")
        self.assertNotIn("Reduce dose", sections[0]["text"])

    def test_long_spl_section_windows_preserve_source_and_overlap(self):
        passages = split_sections([
            {"section_index": 22, "title": "7 DRUG INTERACTIONS",
             "text": "one two three four five six seven eight nine ten"},
        ], max_words=6, overlap=2)
        self.assertEqual([(p.passage_id, p.section_index, p.word_start, p.word_end)
                          for p in passages], [(0, 22, 0, 6), (1, 22, 4, 10)])
        self.assertTrue(passages[0].text.endswith("five six"))
        self.assertTrue(passages[1].text.startswith("five six"))

    def test_qa_score_rejects_citation_outside_evidence(self):
        prompt = {
            "qid": "q1", "task": "factual", "retrieved_indices": [1, 2],
            "gold_indices": [1], "reference_answer": "Ten milligrams",
        }
        response = {"parsed_response": {
            "answer": "Ten milligrams", "citations": ["P0001", "P0099"], "abstain": False,
        }}
        result = score_one(prompt, response)
        self.assertEqual(result["invalid_citation_count"], 1)
        self.assertEqual(result["gold_citation_recall"], 1.0)
        self.assertTrue(result["correct_refusal_decision"])

    def test_qa_score_rejects_nested_citation_arrays(self):
        prompt = {"qid": "q2", "task": "factual", "retrieved_indices": [1], "gold_indices": [1]}
        response = {"parsed_response": {
            "answer": "A", "citations": [["P0001"]], "abstain": False,
        }}
        self.assertFalse(score_one(prompt, response)["valid_schema"])

    def test_claim_level_complete_evidence_accepts_alternates(self):
        prompt = {
            "qid": "q3", "task": "multihop", "schema_version": 2,
            "retrieved_indices": [2, 4], "gold_indices": [1, 2, 3, 4],
            "gold_claims": [[1, 2], [3, 4]], "reference_answer": "A and B",
        }
        response = {"parsed_response": {
            "answer": "A and B", "abstain": False,
            "claims": [
                {"text": "A", "citations": ["P0002"]},
                {"text": "B", "citations": ["P0004"]},
            ],
        }}
        result = score_one(prompt, response)
        self.assertTrue(result["valid_schema"])
        self.assertTrue(result["retrieval_complete_evidence"])
        self.assertTrue(result["cited_complete_evidence"])

    def test_drug_interaction_flag_requires_entity_local_contraindication(self):
        mavacamten_passage = (
            "See Contraindications for some cardiovascular drugs. "
            "Mavacamten: co-administration may increase exposure and heart failure risk. "
            "Discontinue mavacamten during PAXLOVID treatment."
        )
        self.assertEqual(
            contraindication_flag("PAXLOVID is contraindicated with mavacamten", "mavacamten",
                                  [mavacamten_passage]),
            "flag_no_entity_local_contraindication",
        )
        self.assertEqual(
            contraindication_flag("Suzetrigine is contraindicated", "suzetrigine",
                                  ["Coadministration with suzetrigine is contraindicated."]),
            "not_flagged_local_keyword_only",
        )

    def test_spl_table_rows_do_not_bleed_into_adjacent_drugs(self):
        xml = b'''<document xmlns="urn:hl7-org:v3"><component><section>
          <title>7 DRUG INTERACTIONS</title><text>Intro.
          <table><caption>Interactions</caption><thead><tr><th>Class</th><th>Drug</th><th>Comment</th></tr></thead>
          <tbody><tr><td>cardiovascular</td><td>alpha</td><td>Contraindicated.</td></tr>
          <tr><td></td><td>beta</td><td>Interrupt treatment.</td></tr></tbody></table>After table.
          </text></section></component></document>'''
        passages = parse_structured_passages(xml)
        rows = [p for p in passages if p.kind == "table_row"]
        prose = [p for p in passages if p.kind == "prose"]
        self.assertEqual(len(rows), 2)
        self.assertIn("alpha", rows[0].text)
        self.assertTrue(rows[1].inherited_first_cell)
        self.assertIn("Class: cardiovascular", rows[1].text)
        self.assertNotIn("Contraindicated", rows[1].text)
        self.assertNotIn("alpha", prose[0].text)
        self.assertIn("After table", prose[0].text)

    def test_spl_empty_data_row_does_not_inherit_category(self):
        xml = b'''<document xmlns="urn:hl7-org:v3"><component><section><title>Interactions</title>
          <text><table><thead><tr><th>Class</th><th>Drug</th></tr></thead><tbody>
          <tr><td>cardiovascular</td><td>alpha</td></tr>
          <tr><td></td><td></td></tr>
          <tr><td></td><td>beta</td></tr>
          </tbody></table></text></section></component></document>'''
        rows = [p for p in parse_structured_passages(xml) if p.kind == "table_row"]
        self.assertEqual(len(rows), 2)
        self.assertIn("Drug: beta", rows[1].text)
        self.assertTrue(rows[1].inherited_first_cell)

    def test_spl_rowspan_and_colspan_keep_drug_and_column_alignment(self):
        xml = b'''<document xmlns="urn:hl7-org:v3"><section><title>Doses</title><text>
          <table><thead><tr><th>Group</th><th colspan="2">Regimen</th></tr>
          <tr><th>Group</th><th>Day</th><th>Dose</th></tr></thead><tbody>
          <tr><td rowspan="2">Severe impairment</td><td>Day 1</td><td>300 mg</td></tr>
          <tr><td>Day 2</td><td>150 mg</td></tr>
          </tbody></table></text></section></document>'''
        rows = [p for p in parse_structured_passages(xml) if p.kind == "table_row"]
        self.assertEqual(len(rows), 2)
        self.assertIn("Group: Severe impairment", rows[1].text)
        self.assertIn("Regimen / Day: Day 2", rows[1].text)
        self.assertIn("Regimen / Dose: 150 mg", rows[1].text)
        self.assertNotIn("Group: Day 2", rows[1].text)
        self.assertTrue(rows[1].inherited_first_cell)

    def test_version_gate_uses_named_entity_and_full_pinned_label(self):
        self.assertEqual(target_entity("What does coadministration with mavacamten require?"),
                         "mavacamten")
        self.assertTrue(absent_from_label("mavacamten", [{"text": "Other drug interactions."}]))
        self.assertFalse(absent_from_label("mavacamten", [{"text": "Discontinue mavacamten."}]))

    def test_scoring_rejects_response_from_changed_prompt(self):
        prompt = {"qid": "q", "task": "factual", "system_prompt": "rules", "user_prompt": "new question"}
        response = {"qid": "q", "task": "factual",
                    "prompt_sha256": sha256("rulesold question".encode()).hexdigest()}
        with self.assertRaisesRegex(ValueError, "Prompt hash or task mismatch"):
            validate_provenance({"q": prompt}, {"q": response})


if __name__ == "__main__":
    unittest.main()
