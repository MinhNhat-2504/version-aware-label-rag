"""Build version-scoped development prompts with atomic SPL table rows."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .data import Label, read_jsonl
from .prompts import SYSTEM_PROMPT
from .retrieval import LabelIndex
from .spl_tables import parse_structured_passages, structured_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/snapshots/version_pair_paxlovid.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/curated/version_pilot_paxlovid.jsonl"))
    parser.add_argument("--xml-dir", type=Path, default=Path("results/pinned_xml_development"))
    parser.add_argument("--method", choices=("bm25", "evidence_set", "greedy_cover"), default="bm25")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair = next(read_jsonl(args.source))
    versions = {v["spl_version"]: v for v in pair["versions"]}
    indexed = {}
    for version, source in versions.items():
        path = args.xml_dir / f"{pair['set_id']}_v{version}.xml"
        xml_bytes = path.read_bytes()
        if sha256(xml_bytes).hexdigest() != source["xml_sha256"]:
            raise ValueError(f"Pinned XML mismatch: {path}")
        passages = parse_structured_passages(xml_bytes)
        chunks = tuple(p.section_title + "\n" + p.text for p in passages)
        indexed[version] = (passages, LabelIndex(Label(pair["set_id"], pair["title"], chunks)),
                            structured_manifest(xml_bytes, passages))
    output = []
    for question in read_jsonl(args.questions):
        version = question["spl_version"]
        source = versions[version]
        if question["xml_sha256"] != source["xml_sha256"]:
            raise ValueError(f"Question source mismatch: {question['qid']}")
        passages, index, manifest = indexed[version]
        ranking = index.retrieve(question["question"], args.method)
        selected = [passages[position] for position in ranking]
        term = question["absence_check"]["term"].casefold()
        groups = [[p.passage_id for p in passages
                   if p.section_index in claim["acceptable_section_indices"] and term in p.text.casefold()]
                  for claim in question["claims"]]
        if any(not group for group in groups):
            raise ValueError(f"No row/prose evidence for {question['qid']}")
        payload = {
            "qid": question["qid"], "set_id": pair["set_id"], "spl_version": version,
            "xml_sha256": source["xml_sha256"], "task": question["task"],
            "question": question["question"], "reference_answer": question["reference_answer"],
            "schema_version": 2, "passage_scheme": "SPL prose windows plus atomic table rows",
            "parser_version": "row_v3_span_aligned",
            "retrieved_indices": [p.passage_id for p in selected],
            "passage_map": [{"passage_id": p.passage_id, "section_index": p.section_index,
                             "kind": p.kind, "table_index": p.table_index, "row_index": p.row_index}
                            for p in selected],
            "gold_indices": sorted(set().union(*map(set, groups))) if groups else [],
            "gold_claims": groups, "source_manifest": manifest,
            "system_prompt": SYSTEM_PROMPT +
                f"\nUse only the passages from SPL version {version}. If this version does not state an interaction for a named drug, abstain; absence is not evidence of safety.",
            "user_prompt": f"Drug: {pair['title']}\nTarget SPL version: {version}\n"
                           f"Question: {question['question']}\n\nLabel passages:\n" +
                           "\n\n".join(
                               f"[P{p.passage_id:04d}]\nSPL section {p.section_index}: {p.section_title}"
                               f" ({'table row ' + str(p.table_index) + ':' + str(p.row_index) if p.kind == 'table_row' else 'prose'})\n{p.text}"
                               for p in selected),
        }
        output.append(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "method": args.method,
                      "items": len(output),
                      "answerable_complete_retrieval": sum(all(set(group) & set(row["retrieved_indices"])
                                                           for group in row["gold_claims"])
                                                           for row in output if row["gold_claims"]),
                      "mean_prompt_words": round(sum(len(row["user_prompt"].split()) for row in output) / len(output), 1)}))


if __name__ == "__main__":
    main()
