"""Print compact word-level source deltas for version-pair annotation."""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

from .data import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--max-sections", type=int, default=5)
    parser.add_argument("--max-edits", type=int, default=6)
    args = parser.parse_args()
    for row in read_jsonl(args.sources):
        old, new = row["versions"]
        old_sections = {(s["code"], s["title"]): s["text"] for s in old["sections"]}
        print(f"\n{row['set_id']} {row['title']} v{old['spl_version']}→v{new['spl_version']}")
        changed = row.get("clinical_change_sections") or []
        for item in changed[:args.max_sections]:
            section = next(s for s in new["sections"] if s["section_index"] == item["section_index"])
            before = old_sections.get((section["code"], section["title"]), "").split()
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
