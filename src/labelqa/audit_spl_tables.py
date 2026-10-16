"""Audit row-preserving SPL parsing on previously used development XMLs."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .data import read_jsonl
from .recent import NS
from .spl_tables import (flattened_text, local_name, parse_structured_passages,
                         structured_manifest, table_layout_rows)


SOURCES = (Path("data/snapshots/version_pair_paxlovid.jsonl"),
           Path("data/snapshots/version_pairs_3.jsonl"))
XML_DIR = Path("results/pinned_xml_development")


def main() -> None:
    rows = []
    for snapshot in SOURCES:
        for pair in read_jsonl(snapshot):
            for version in pair["versions"]:
                path = XML_DIR / f"{pair['set_id']}_v{version['spl_version']}.xml"
                xml_bytes = path.read_bytes()
                if sha256(xml_bytes).hexdigest() != version["xml_sha256"]:
                    raise ValueError(f"Pinned XML mismatch: {path}")
                root = ET.fromstring(xml_bytes)
                sections = root.findall(".//hl7:section", NS)
                tables = [t for section in sections
                          for text in [section.find("hl7:text", NS)] if text is not None
                          for t in text.findall(".//hl7:table", NS)]
                source_rows = 0
                mixed_heading_rows = 0
                nested_tables = 0
                layout_counts = Counter()
                layout_examples = []
                for section_index, section in enumerate(sections):
                    text = section.find("hl7:text", NS)
                    if text is None:
                        continue
                    for table_index, table in enumerate(text.findall(".//hl7:table", NS)):
                        for layout in table_layout_rows(table):
                            cells = layout["cells"]
                            if not cells:
                                continue
                            layout_counts["rows"] += 1
                            layout_counts["cells_with_colspan"] += sum(c["colspan"] > 1 for c in cells)
                            layout_counts["cells_with_rowspan"] += sum(c["rowspan"] > 1 for c in cells)
                            has_data = any(c["tag"] == "td" and c["text"] for c in cells)
                            if not has_data:
                                continue
                            carried = bool(layout["carried_columns"])
                            shifted = any(c["tag"] == "td" and c["text"] and
                                          c["column"] != c["ordinal"] for c in cells)
                            merged = any(c["colspan"] > 1 for c in cells)
                            layout_counts["nonempty_data_rows"] += 1
                            layout_counts["data_rows_with_carried_rowspan"] += int(carried)
                            layout_counts["data_rows_with_shifted_cell_position"] += int(shifted)
                            layout_counts["data_rows_with_colspan"] += int(merged)
                            layout_counts["data_rows_with_layout_risk"] += int(carried or shifted or merged)
                            if (carried or shifted or merged) and len(layout_examples) < 8:
                                layout_examples.append({
                                    "section_index": section_index, "table_index": table_index,
                                    "row_index": layout["row_index"],
                                    "carried_columns": layout["carried_columns"],
                                    "cell_columns": [c["column"] for c in cells],
                                    "cell_ordinals": [c["ordinal"] for c in cells],
                                    "cell_text_prefixes": [c["text"][:100] for c in cells],
                                })
                for table in tables:
                    nested_tables += len(table.findall(".//hl7:table", NS))
                    for tr in table.findall(".//hl7:tr", NS):
                        cells = [cell for cell in tr if local_name(cell.tag) in ("td", "th")]
                        tags = {local_name(cell.tag) for cell in cells}
                        mixed_heading_rows += tags == {"td", "th"}
                        if "td" in tags and any(flattened_text(cell) for cell in cells):
                            source_rows += 1
                passages = parse_structured_passages(xml_bytes)
                manifest = structured_manifest(xml_bytes, passages)
                row_passages = [p for p in passages if p.kind == "table_row"]
                rows.append({
                    "set_id": pair["set_id"], "title": pair["title"],
                    "spl_version": version["spl_version"], "xml_sha256": version["xml_sha256"],
                    "xml_section_count": len(sections), "snapshot_nonempty_sections": len(version["sections"]),
                    "table_count": len(tables), "source_nonempty_data_rows": source_rows,
                    "emitted_row_passages": len(row_passages),
                    "mixed_th_td_rows": mixed_heading_rows, "nested_tables": nested_tables,
                    "layout_counts": dict(layout_counts),
                    "layout_risk_examples": layout_examples,
                    "inherited_first_cell_rows": sum(p.inherited_first_cell for p in row_passages),
                    "max_row_words": max((len(p.text.split()) for p in row_passages), default=0),
                    "passage_manifest": manifest,
                })
    report = {"status": "automatic_structure_audit_development_labels_only",
              "versions": len(rows), "labels": len({r["set_id"] for r in rows}),
              "all_nonempty_source_data_rows_emitted": all(r["source_nonempty_data_rows"] == r["emitted_row_passages"] for r in rows),
              "layout_totals": dict(sum((Counter(row["layout_counts"]) for row in rows), Counter())),
              "rows": rows,
              "limitation": "Merged-cell counts identify rows requiring span-aware rendering. Logical coordinates are checked mechanically, but clinical meaning, footnote/data-row semantics and citation support still need source-layout inspection."}
    output = Path("results/spl_table_structure_audit.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"versions": len(rows), "labels": report["labels"],
                      "table_count": sum(r["table_count"] for r in rows),
                      "source_rows": sum(r["source_nonempty_data_rows"] for r in rows),
                      "emitted_rows": sum(r["emitted_row_passages"] for r in rows),
                      "mixed_th_td_rows": sum(r["mixed_th_td_rows"] for r in rows),
                      "nested_tables": sum(r["nested_tables"] for r in rows),
                      "layout_totals": report["layout_totals"],
                      "all_rows_emitted": report["all_nonempty_source_data_rows_emitted"],
                      "output": str(output)}))


if __name__ == "__main__":
    main()
