"""Assistant-drafted version-change QA from the frozen PAXLOVID SPL pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import read_jsonl


QUESTIONS = (
    ("suzetrigine", "What does this PAXLOVID label say about coadministration with suzetrigine, and why?",
     "Coadministration is contraindicated because of potential serious or life-threatening suzetrigine adverse reactions.",
     [("Coadministration contraindication", [11, 22]), ("Reason for contraindication", [22])]),
    ("mavacamten", "What does this PAXLOVID label say about coadministration with mavacamten, including risk and interruption?",
     "Increased mavacamten exposure may increase heart failure risk. Discontinue mavacamten during PAXLOVID treatment; resumption within five days may increase exposure.",
     [("Increased exposure and heart failure risk", [22]), ("Interruption and resumption guidance", [22])]),
)


def build(snapshot: dict) -> list[dict]:
    versions = {row["spl_version"]: row for row in snapshot["versions"]}
    if set(versions) != {11, 12}:
        raise ValueError("Expected frozen PAXLOVID versions 11 and 12")
    records = []
    for n, (term, question, answer, claims) in enumerate(QUESTIONS, 1):
        for version in (11, 12):
            source = versions[version]
            present = {section["section_index"] for section in source["sections"]}
            occurrences = sum(term.casefold() in section["text"].casefold()
                              for section in source["sections"])
            if (version == 11 and occurrences) or (version == 12 and not occurrences):
                raise ValueError(f"Expected version change not found for {term} v{version}")
            if version == 12 and any(not set(indices) <= present for _, indices in claims):
                raise ValueError(f"Missing source section for {term}")
            records.append({
                "qid": f"version-{n:02d}-v{version}",
                "set_id": snapshot["set_id"], "spl_version": version,
                "published_date": source["published_date"],
                "xml_sha256": source["xml_sha256"], "source_url": source["source_url"],
                "question": question, "task": "factual" if version == 12 else "refusal",
                "answerable_from_version": version == 12,
                "reference_answer": answer if version == 12 else
                    f"The pinned version 11 label does not mention {term}; abstain from inferring compatibility.",
                "claims": ([{"claim": text, "acceptable_section_indices": indices}
                            for text, indices in claims] if version == 12 else []),
                "absence_check": {"term": term, "full_label_occurrences": occurrences},
                "annotation_status": "assistant_draft_pending_independent_expert_review",
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/snapshots/version_pair_paxlovid.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/curated/version_pilot_paxlovid.jsonl"))
    args = parser.parse_args()
    rows = list(read_jsonl(args.source))
    if len(rows) != 1:
        raise ValueError("Expected one PAXLOVID version pair")
    records = build(rows[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "questions": len(records)}))


if __name__ == "__main__":
    main()
