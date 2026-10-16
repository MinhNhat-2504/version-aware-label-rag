"""Run a dependency-free synthetic smoke test through the core QA pipeline."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import Label
from .prompts import make_prompt
from .retrieval import LabelIndex
from .score_qa import score_one, validate_provenance
from .spl_tables import parse_structured_passages, structured_manifest


SYNTHETIC_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <component><structuredBody>
    <component><section>
      <title>INDICATIONS AND USAGE</title>
      <text><paragraph>SYNTH-101 is indicated to treat Condition Alpha in adults.</paragraph></text>
    </section></component>
    <component><section>
      <title>DOSAGE AND ADMINISTRATION</title>
      <text>
        <table>
          <caption>Administration schedule</caption>
          <thead><tr><th>Population</th><th>Schedule</th></tr></thead>
          <tbody><tr><td>Adults</td><td>One synthetic unit once daily.</td></tr></tbody>
        </table>
      </text>
    </section></component>
    <component><section>
      <title>STORAGE AND HANDLING</title>
      <text><paragraph>Store the synthetic package at controlled room temperature.</paragraph></text>
    </section></component>
  </structuredBody></component>
</ClinicalDocument>
"""

QUESTION = (
    "What condition and population is SYNTH-101 indicated for, and what administration "
    "schedule is listed for that population?"
)
REFERENCE_ANSWER = (
    "SYNTH-101 is indicated for Condition Alpha in adults, with one synthetic unit once daily."
)
METHODS = ("bm25", "tfidf", "rrf_lexical", "ontology_cover", "evidence_set", "greedy_cover")


def run_demo() -> dict:
    passages = parse_structured_passages(SYNTHETIC_XML)
    if len(passages) != 3 or sum(p.kind == "table_row" for p in passages) != 1:
        raise RuntimeError("Synthetic SPL parser output changed unexpectedly")
    label = Label(
        set_id="00000000-0000-0000-0000-000000000101",
        drug_name="SYNTH-101",
        chunks=tuple(f"{passage.section_title}\n{passage.text}" for passage in passages),
    )
    index = LabelIndex(label)
    rankings = {method: index.retrieve(QUESTION, method=method, k=3) for method in METHODS}
    required = [{0}, {1}]
    coverage = {
        method: all(set(ranking) & group for group in required)
        for method, ranking in rankings.items()
    }

    selected_method = "evidence_set"
    ranking = rankings[selected_method]
    rendered = make_prompt(label, QUESTION, ranking)
    prompt = {
        "qid": "synthetic-multihop-001",
        "set_id": label.set_id,
        "task": "multihop",
        "question": QUESTION,
        "reference_answer": REFERENCE_ANSWER,
        "schema_version": 2,
        "retrieved_indices": ranking,
        "gold_indices": [0, 1],
        "gold_claims": [[0], [1]],
        **rendered,
    }
    prompt_hash = sha256((prompt["system_prompt"] + prompt["user_prompt"]).encode()).hexdigest()
    response = {
        "qid": prompt["qid"],
        "task": prompt["task"],
        "prompt_sha256": prompt_hash,
        "parsed_response": {
            "answer": REFERENCE_ANSWER,
            "abstain": False,
            "claims": [
                {"text": "SYNTH-101 is indicated for Condition Alpha in adults.",
                 "citations": ["P0000"]},
                {"text": "The adult schedule is one synthetic unit once daily.",
                 "citations": ["P0001"]},
            ],
        },
    }
    validate_provenance({prompt["qid"]: prompt}, {response["qid"]: response})
    score = score_one(prompt, response)
    if not (score.get("valid_schema") and score.get("retrieval_complete_evidence")
            and score.get("cited_complete_evidence") and score.get("invalid_citation_count") == 0):
        raise RuntimeError("Synthetic prompt/citation smoke test failed")

    return {
        "status": "synthetic_smoke_passed",
        "contains_real_drug_or_holdout_data": False,
        "model_called": False,
        "xml_sha256": sha256(SYNTHETIC_XML).hexdigest(),
        "parser": structured_manifest(SYNTHETIC_XML, passages),
        "question": QUESTION,
        "passages": [
            {
                "passage_id": passage.passage_id,
                "section_index": passage.section_index,
                "section_title": passage.section_title,
                "kind": passage.kind,
                "table_index": passage.table_index,
                "row_index": passage.row_index,
                "text": passage.text,
            }
            for passage in passages
        ],
        "retrieval": {
            method: {"ranking": rankings[method], "complete_evidence": coverage[method]}
            for method in METHODS
        },
        "selected_method": selected_method,
        "prompt_sha256": prompt_hash,
        "mechanical_score": score,
        "interpretation": (
            "Software smoke test only. The response is constructed from synthetic source text; "
            "no model quality or clinical correctness is measured."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    report = run_demo()
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
