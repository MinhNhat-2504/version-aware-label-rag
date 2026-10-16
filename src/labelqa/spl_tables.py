"""Preserve SPL table-row boundaries while splitting non-table narrative."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import xml.etree.ElementTree as ET

from .recent import NS
from .spl_passages import split_sections


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def flattened_text(node: ET.Element) -> str:
    return " ".join(" ".join(node.itertext()).split())


def narrative_text(node: ET.Element) -> str:
    """Extract all visible text except entire table subtrees."""
    parts: list[str] = []

    def visit(current: ET.Element) -> None:
        if current.text:
            parts.append(current.text)
        for child in current:
            if local_name(child.tag) != "table":
                visit(child)
            if child.tail:
                parts.append(child.tail)

    visit(node)
    return " ".join(" ".join(parts).split())


def table_layout_rows(table: ET.Element) -> list[dict]:
    """Expand XML cell spans into logical columns while retaining source row IDs."""
    if table.findall(".//hl7:table", NS):
        raise ValueError("Nested SPL tables need separate layout handling")
    carry: dict[int, dict] = {}
    results = []
    for row_index, row in enumerate(table.findall(".//hl7:tr", NS)):
        cells = [cell for cell in row if local_name(cell.tag) in ("td", "th")]
        occupied = set(carry)
        next_carry: dict[int, dict] = {}
        placements = []
        column = 0
        for ordinal, cell in enumerate(cells):
            try:
                colspan = int(cell.get("colspan", "1"))
                rowspan = int(cell.get("rowspan", "1"))
            except ValueError as exc:
                raise ValueError(f"Invalid SPL table span at row {row_index}") from exc
            if colspan < 1 or rowspan < 1:
                raise ValueError(f"Invalid SPL table span at row {row_index}")
            while any(position in occupied for position in range(column, column + colspan)):
                column += 1
            placement = {"tag": local_name(cell.tag), "ordinal": ordinal,
                         "column": column, "colspan": colspan,
                         "rowspan": rowspan, "text": flattened_text(cell)}
            placements.append(placement)
            for position in range(column, column + colspan):
                occupied.add(position)
                if rowspan > 1:
                    next_carry[position] = {**placement, "remaining": rowspan - 1}
            column += colspan
        carried_cells = {cell["column"]: cell for cell in carry.values()}.values()
        results.append({"row_index": row_index, "carried_columns": sorted(carry),
                        "carried_cells": list(carried_cells), "cells": placements})
        carry = {position: {**cell, "remaining": cell["remaining"] - 1}
                 for position, cell in carry.items() if cell["remaining"] > 1}
        carry.update(next_carry)
    return results


@dataclass(frozen=True)
class StructuredPassage:
    passage_id: int
    section_index: int
    section_title: str
    kind: str
    text: str
    table_index: int | None = None
    row_index: int | None = None
    inherited_first_cell: bool = False


def parse_structured_passages(xml_bytes: bytes, max_words: int = 160, overlap: int = 30) -> list[StructuredPassage]:
    root = ET.fromstring(xml_bytes)
    passages: list[StructuredPassage] = []
    for section_index, section in enumerate(root.findall(".//hl7:section", NS)):
        text_node = section.find("hl7:text", NS)
        if text_node is None:
            continue
        title_node = section.find("hl7:title", NS)
        section_title = flattened_text(title_node) if title_node is not None else ""
        prose = narrative_text(text_node)
        if prose:
            windows = split_sections([{"section_index": section_index, "title": section_title,
                                       "text": prose}], max_words=max_words, overlap=overlap)
            for window in windows:
                passages.append(StructuredPassage(len(passages), section_index, section_title,
                                                  "prose", window.text))
        for table_index, table in enumerate(text_node.findall(".//hl7:table", NS)):
            caption_node = table.find("hl7:caption", NS)
            caption = flattened_text(caption_node) if caption_node is not None else ""
            headers: dict[int, str] = {}
            previous_was_header = False
            first_cell_context = ""
            for layout in table_layout_rows(table):
                row_index = layout["row_index"]
                cells = layout["cells"]
                if not cells:
                    continue
                if any(cell["tag"] == "th" for cell in cells):
                    if not previous_was_header:
                        headers = {}
                    for cell in cells:
                        if cell["tag"] != "th" or not cell["text"]:
                            continue
                        for column in range(cell["column"], cell["column"] + cell["colspan"]):
                            previous = headers.get(column, "")
                            if cell["text"] not in previous:
                                headers[column] = (previous + " / " if previous else "") + cell["text"]
                    first_cell_context = ""
                    previous_was_header = True
                    continue
                previous_was_header = False
                if not any(cell["text"] for cell in cells):
                    continue
                carried = [cell for cell in layout["carried_cells"]
                           if cell["tag"] == "td" and cell["text"]]
                inherited = any(cell["column"] == 0 for cell in carried)
                rendered = list(carried)
                for cell in cells:
                    value = cell["text"]
                    if cell["column"] == 0 and cell["colspan"] == 1:
                        if value:
                            first_cell_context = value
                        elif first_cell_context:
                            value = first_cell_context
                            inherited = True
                    if value:
                        rendered.append({**cell, "text": value})
                fields = []
                for cell in sorted(rendered, key=lambda x: x["column"]):
                    columns = range(cell["column"], cell["column"] + cell["colspan"])
                    labels = list(dict.fromkeys(headers.get(i, f"Column {i + 1}") for i in columns))
                    fields.append(f"{' / '.join(labels)}: {cell['text']}")
                if not fields:
                    continue
                row_text = (caption + ". " if caption else "") + " | ".join(fields)
                passages.append(StructuredPassage(
                    len(passages), section_index, section_title, "table_row", row_text,
                    table_index, row_index, inherited,
                ))
    return passages


def structured_manifest(xml_bytes: bytes, passages: list[StructuredPassage]) -> dict:
    return {"xml_sha256": sha256(xml_bytes).hexdigest(), "passage_count": len(passages),
            "prose_count": sum(p.kind == "prose" for p in passages),
            "table_row_count": sum(p.kind == "table_row" for p in passages)}
