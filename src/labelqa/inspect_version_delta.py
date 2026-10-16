"""Print compact word-level source deltas for version-pair annotation."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

from .data import read_jsonl


def occurrence_keys(sections: list[dict]) -> list[tuple[str, str, int]]:
    """Key sections by code, title and occurrence, so repeated titles never overwrite.

    Labels repeat subsection titles such as "Risk Summary" under 8.1 Pregnancy and 8.2
    Lactation; a plain (code, title) key would compare the wrong subsections.
    """
    seen: dict[tuple[str, str], int] = {}
    keys = []
    for section in sections:
        base = (section["code"], section["title"])
        keys.append((*base, seen.get(base, 0)))
        seen[base] = seen.get(base, 0) + 1
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--max-sections", type=int, default=5)
    parser.add_argument("--max-edits", type=int, default=6)
    args = parser.parse_args()
    for row in read_jsonl(args.sources):
        old, new = row["versions"]
        old_sections = {key: s["text"] for key, s in zip(occurrence_keys(old["sections"]), old["sections"])}
        new_keys = dict(zip((s["section_index"] for s in new["sections"]), occurrence_keys(new["sections"])))
        print(f"\n{row['set_id']} {row['title']} v{old['spl_version']}→v{new['spl_version']}")
        changed = row.get("clinical_change_sections") or []
        for item in changed[:args.max_sections]:
            section = next(s for s in new["sections"] if s["section_index"] == item["section_index"])
            before = old_sections.get(new_keys[section["section_index"]], "").split()
            after = section["text"].split()
            edits = []
            for tag, i, j, k, l in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
                if tag != "equal":
                    edits.append((" ".join(before[i:j])[:160], " ".join(after[k:l])[:160]))
            print(f"  {section['title'][:72]}: {len(edits)} word edits")
            for prior, current in edits[:args.max_edits]:
                print(f"    − {prior}\n    + {current}")


if __name__ == "__main__":
    main()
