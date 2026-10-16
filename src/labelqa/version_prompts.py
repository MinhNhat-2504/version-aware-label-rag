"""Prepare version-scoped PAXLOVID pilot prompts without mixing SPL revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Label, read_jsonl
from .prompts import SYSTEM_PROMPT
from .retrieval import LabelIndex
from .spl_passages import split_sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/snapshots/version_pair_paxlovid.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/curated/version_pilot_paxlovid.jsonl"))
    parser.add_argument("--method", choices=["bm25", "evidence_set"], default="bm25")
    parser.add_argument("--chunk-words", type=int, default=0,
                        help="Split long sections into overlapping word windows; 0 keeps whole sections")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = list(read_jsonl(args.source))
    if len(pairs) != 1:
        raise ValueError("Expected one frozen version pair")
    pair = pairs[0]
    versions = {row["spl_version"]: row for row in pair["versions"]}
    indexes = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for question in read_jsonl(args.questions):
            version = question["spl_version"]
            source = versions[version]
            if (question["set_id"] != pair["set_id"] or
                    question["xml_sha256"] != source["xml_sha256"]):
                raise ValueError(f"Version/source mismatch for {question['qid']}")
            sections = source["sections"]
            if version not in indexes:
                if args.chunk_words:
                    passages = split_sections(sections, max_words=args.chunk_words)
                else:
                    from .spl_passages import Passage
                    passages = [Passage(section["section_index"], section["section_index"],
                                        section["title"], section["text"], 0,
                                        len(section["text"].split())) for section in sections]
                chunks = tuple(passage.title + "\n" + passage.text for passage in passages)
                indexes[version] = LabelIndex(Label(pair["set_id"], pair["title"], chunks))
                indexes[(version, "passages")] = passages
            passages = indexes[(version, "passages")]
            ranking = indexes[version].retrieve(question["question"], args.method)
            selected = [passages[position] for position in ranking]
            term = question["absence_check"]["term"].casefold()
            groups = [
                [passage.passage_id for passage in passages
                 if passage.section_index in claim["acceptable_section_indices"]
                 and term in passage.text.casefold()]
                for claim in question["claims"]
            ]
            if any(not group for group in groups):
                raise ValueError(f"No supporting passage window for {question['qid']}")
            payload = {
                "qid": question["qid"], "set_id": pair["set_id"],
                "spl_version": version, "xml_sha256": source["xml_sha256"],
                "task": question["task"], "question": question["question"],
                "reference_answer": question["reference_answer"], "schema_version": 2,
                "retrieved_indices": [passage.passage_id for passage in selected],
                "passage_map": [{"passage_id": passage.passage_id,
                                 "section_index": passage.section_index,
                                 "word_start": passage.word_start, "word_end": passage.word_end}
                                for passage in selected],
                "gold_indices": sorted(set().union(*map(set, groups))) if groups else [],
                "gold_claims": groups,
                "system_prompt": SYSTEM_PROMPT +
                    f"\nUse only the passages from SPL version {version}. If this version does not state an interaction for a named drug, abstain; absence is not evidence of safety.",
                "user_prompt": f"Drug: {pair['title']}\nTarget SPL version: {version}\n"
                               f"Question: {question['question']}\n\nLabel passages:\n" +
                               "\n\n".join(f"[P{passage.passage_id:04d}]\n"
                                           f"SPL section {passage.section_index}: {passage.title}\n{passage.text}"
                                           for passage in selected),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "method": args.method}))


if __name__ == "__main__":
    main()
