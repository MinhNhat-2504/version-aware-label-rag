"""Generate end-to-end prompts for the assistant-drafted DailyMed pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Label, read_jsonl
from .prompts import SYSTEM_PROMPT
from .retrieval import LabelIndex


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path("data/snapshots/recent_label_pool_10.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/curated/temporal_pilot.jsonl"))
    parser.add_argument("--method", choices=["bm25", "ontology_cover", "evidence_set"], default="bm25")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = {row["set_id"]: row for row in read_jsonl(args.pool)}
    questions = list(read_jsonl(args.questions))
    indexes = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for question in questions:
            label = pool[question["set_id"]]
            if (question["source_sha256"] != label["source_sha256"] or
                    question["spl_version"] != label["spl_version"]):
                raise ValueError(f"Changed source version: {question['qid']}")
            sections = label["sections"]
            if question["set_id"] not in indexes:
                chunks = tuple(section["title"] + "\n" + section["text"] for section in sections)
                indexes[question["set_id"]] = LabelIndex(Label(
                    question["set_id"], label["title"], chunks
                ))
            ranking = indexes[question["set_id"]].retrieve(question["question"], args.method)
            retrieved = [sections[position]["section_index"] for position in ranking]
            passages = "\n\n".join(
                f"[P{sections[position]['section_index']:04d}]\n"
                f"{sections[position]['title']}\n{sections[position]['text']}"
                for position in ranking
            )
            groups = [claim["acceptable_section_indices"] for claim in question["claims"]]
            payload = {
                "qid": question["qid"], "set_id": question["set_id"],
                "spl_version": question["spl_version"],
                "source_sha256": question["source_sha256"],
                "annotation_status": question["annotation_status"],
                "task": question["task"], "question": question["question"],
                "reference_answer": question["reference_answer"],
                "schema_version": 2, "retrieved_indices": retrieved,
                "gold_indices": sorted(set().union(*map(set, groups))),
                "gold_claims": groups,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": f"Drug: {label['title']}\nSPL version: {label['spl_version']}\n"
                               f"Question: {question['question']}\n\nLabel passages:\n{passages}",
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "questions": len(questions), "method": args.method}))


if __name__ == "__main__":
    main()
