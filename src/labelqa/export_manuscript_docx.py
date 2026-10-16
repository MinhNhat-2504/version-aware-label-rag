"""Export the working Markdown manuscript as an editable Word document.

This uses only the Python standard library and keeps Markdown as the paper's
single editable source. It supports the headings, tables, emphasis, code spans,
and links used by MANUSCRIPT_DRAFT.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
XSI = "http://www.w3.org/2001/XMLSchema-instance"
ET.register_namespace("w", W)
ET.register_namespace("r", R)
ET.register_namespace("cp", CP)
ET.register_namespace("dc", DC)
ET.register_namespace("dcterms", DCTERMS)
ET.register_namespace("xsi", XSI)

INLINE = re.compile(r"(\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|`[^`]+`|(?<!\*)\*[^*]+\*(?!\*))")
TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")


def tag(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def add(parent: ET.Element, name: str, attrs: dict[str, str] | None = None,
        value: str | None = None) -> ET.Element:
    node = ET.SubElement(parent, tag(W, name),
                         {tag(W, key): val for key, val in (attrs or {}).items()})
    if value is not None:
        node.text = value
    return node


def xml(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


class WordBuilder:
    def __init__(self) -> None:
        self.document = ET.Element(tag(W, "document"))
        self.body = add(self.document, "body")
        self.links: list[tuple[str, str]] = []

    def paragraph(self, style: str = "Normal") -> ET.Element:
        paragraph = add(self.body, "p")
        properties = add(paragraph, "pPr")
        add(properties, "pStyle", {"val": style})
        return paragraph

    def run(self, parent: ET.Element, value: str, *, bold: bool = False,
            italic: bool = False, code: bool = False) -> None:
        if not value:
            return
        run = add(parent, "r")
        if bold or italic or code:
            properties = add(run, "rPr")
            if bold:
                add(properties, "b")
            if italic:
                add(properties, "i")
            if code:
                add(properties, "rFonts", {"ascii": "Consolas", "hAnsi": "Consolas"})
                add(properties, "sz", {"val": "19"})
        text = add(run, "t", value=value)
        if value.startswith(" ") or value.endswith(" "):
            text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    def inline(self, parent: ET.Element, content: str) -> None:
        position = 0
        for match in INLINE.finditer(content):
            self.run(parent, content[position:match.start()])
            token = match.group()
            if token.startswith("["):
                label, target = token[1:].split("](", 1)
                target = target[:-1]
                rel_id = f"rId{len(self.links) + 3}"
                self.links.append((rel_id, target))
                hyperlink = add(parent, "hyperlink")
                hyperlink.set(tag(R, "id"), rel_id)
                run = add(hyperlink, "r")
                properties = add(run, "rPr")
                add(properties, "color", {"val": "175A93"})
                add(properties, "u", {"val": "single"})
                add(run, "t", value=label)
            elif token.startswith("**"):
                self.run(parent, token[2:-2], bold=True)
            elif token.startswith("*"):
                self.run(parent, token[1:-1], italic=True)
            else:
                self.run(parent, token[1:-1], code=True)
            position = match.end()
        self.run(parent, content[position:])

    def table(self, lines: list[str]) -> None:
        rows = [[cell.strip() for cell in line.strip().strip("|").split("|")]
                for line in lines if not TABLE_RULE.match(line)]
        if not rows:
            return
        columns = len(rows[0])
        if any(len(row) != columns for row in rows):
            raise ValueError("Markdown table has inconsistent column counts")
        table = add(self.body, "tbl")
        properties = add(table, "tblPr")
        add(properties, "tblW", {"w": "9360", "type": "dxa"})
        borders = add(properties, "tblBorders")
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            add(borders, side, {"val": "single", "sz": "4", "color": "B8C4CF"})
        grid = add(table, "tblGrid")
        widths = ([int(9360 * .4)] + [int(9360 * .6 / (columns - 1))] * (columns - 1)
                  if columns > 1 else [9360])
        for width in widths:
            add(grid, "gridCol", {"w": str(width)})
        for index, row in enumerate(rows):
            tr = add(table, "tr")
            if index == 0:
                add(add(tr, "trPr"), "tblHeader", {"val": "1"})
            for cell_index, content in enumerate(row):
                tc = add(tr, "tc")
                tcpr = add(tc, "tcPr")
                add(tcpr, "tcW", {"w": str(widths[cell_index]), "type": "dxa"})
                if index == 0:
                    add(tcpr, "shd", {"fill": "EDF2F6"})
                paragraph = add(tc, "p")
                add(add(paragraph, "pPr"), "pStyle", {"val": "TableText"})
                if index == 0:
                    self.run(paragraph, re.sub(r"`([^`]+)`", r"\1", content), bold=True)
                else:
                    self.inline(paragraph, content)

    def finish(self) -> None:
        section = add(self.body, "sectPr")
        footer_ref = add(section, "footerReference", {"type": "default"})
        footer_ref.set(tag(R, "id"), "rId2")
        add(section, "pgSz", {"w": "12240", "h": "15840"})
        add(section, "pgMar", {"top": "1200", "right": "1320", "bottom": "1200",
                               "left": "1320", "header": "600", "footer": "600", "gutter": "0"})


def parse_markdown(source: str) -> WordBuilder:
    builder = WordBuilder()
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            builder.table(table_lines)
            continue
        if line.startswith("#"):
            match = re.match(r"^(#{1,3})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                paragraph = builder.paragraph("Title" if level == 1 else f"Heading{level - 1}")
                builder.inline(paragraph, match.group(2))
                index += 1
                continue
        text_lines = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|")):
            text_lines.append(lines[index].strip())
            index += 1
        paragraph = builder.paragraph("Normal")
        builder.inline(paragraph, " ".join(text_lines))
    builder.finish()
    return builder


def styles_xml() -> bytes:
    root = ET.Element(tag(W, "styles"))
    defaults = add(root, "docDefaults")
    run_defaults = add(defaults, "rPrDefault")
    props = add(run_defaults, "rPr")
    add(props, "rFonts", {"ascii": "Times New Roman", "hAnsi": "Times New Roman"})
    add(props, "sz", {"val": "22"})
    paragraph_defaults = add(add(defaults, "pPrDefault"), "pPr")
    add(paragraph_defaults, "spacing", {"after": "120", "line": "276", "lineRule": "auto"})
    for ident, name, size, before, after, bold in (
            ("Normal", "Normal", 22, 0, 120, False),
            ("Title", "Title", 34, 0, 260, True),
            ("Heading1", "Heading 1", 28, 300, 140, True),
            ("Heading2", "Heading 2", 24, 220, 100, True),
            ("TableText", "Table Text", 18, 0, 0, False)):
        style = add(root, "style", {"type": "paragraph", "styleId": ident})
        add(style, "name", {"val": name})
        if ident != "Normal":
            add(style, "basedOn", {"val": "Normal"})
        ppr = add(style, "pPr")
        add(ppr, "spacing", {"before": str(before), "after": str(after),
                             "line": "260", "lineRule": "auto"})
        if ident.startswith("Heading"):
            add(ppr, "keepNext")
        rpr = add(style, "rPr")
        add(rpr, "sz", {"val": str(size)})
        if bold:
            add(rpr, "b")
    return xml(root)


def footer_xml() -> bytes:
    root = ET.Element(tag(W, "ftr"))
    paragraph = add(root, "p")
    add(add(paragraph, "pPr"), "jc", {"val": "center"})
    run = add(paragraph, "r")
    add(run, "t", value="Working manuscript  •  ")
    run = add(paragraph, "r")
    add(run, "fldChar", {"fldCharType": "begin"})
    run = add(paragraph, "r")
    instruction = add(run, "instrText", value=" PAGE ")
    instruction.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run = add(paragraph, "r")
    add(run, "fldChar", {"fldCharType": "end"})
    return xml(root)


def build_docx_bytes(source: str) -> bytes:
    builder = parse_markdown(source)
    rels = ET.Element(tag(P, "Relationships"))
    ET.SubElement(rels, tag(P, "Relationship"), {
        "Id": "rId1", "Type": R + "/styles", "Target": "styles.xml"})
    ET.SubElement(rels, tag(P, "Relationship"), {
        "Id": "rId2", "Type": R + "/footer", "Target": "footer1.xml"})
    for ident, target in builder.links:
        ET.SubElement(rels, tag(P, "Relationship"), {
            "Id": ident, "Type": R + "/hyperlink", "Target": target, "TargetMode": "External"})
    content_types = ET.Element(tag(CT, "Types"))
    ET.SubElement(content_types, tag(CT, "Default"), {
        "Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(content_types, tag(CT, "Default"), {
        "Extension": "xml", "ContentType": "application/xml"})
    for part, content_type in (("/word/document.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"),
                               ("/word/styles.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"),
                               ("/word/footer1.xml", "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"),
                               ("/docProps/core.xml", "application/vnd.openxmlformats-package.core-properties+xml")):
        ET.SubElement(content_types, tag(CT, "Override"), {
            "PartName": part, "ContentType": content_type})
    package_rels = ET.Element(tag(P, "Relationships"))
    ET.SubElement(package_rels, tag(P, "Relationship"), {
        "Id": "rId1", "Type": R + "/officeDocument", "Target": "word/document.xml"})
    ET.SubElement(package_rels, tag(P, "Relationship"), {
        "Id": "rId2", "Type": "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties",
        "Target": "docProps/core.xml"})
    core = ET.Element(tag(CP, "coreProperties"))
    ET.SubElement(core, tag(DC, "title")).text = "Evidence-Complete and Version-Aware Question Answering over FDA Drug Labels"
    ET.SubElement(core, tag(DC, "creator")).text = "ADMET research project"
    ET.SubElement(core, tag(DC, "description")).text = "Working manuscript; preliminary results only"
    modified = ET.SubElement(core, tag(DCTERMS, "modified"))
    modified.set(tag(XSI, "type"), "dcterms:W3CDTF")
    modified.text = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for path, data in (("[Content_Types].xml", xml(content_types)),
                           ("_rels/.rels", xml(package_rels)),
                           ("docProps/core.xml", xml(core)),
                           ("word/document.xml", xml(builder.document)),
                           ("word/styles.xml", styles_xml()),
                           ("word/footer1.xml", footer_xml()),
                           ("word/_rels/document.xml.rels", xml(rels))):
            archive.writestr(path, data)
    return buffer.getvalue()


def export(source: Path, output: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    document = build_docx_bytes(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("paper/MANUSCRIPT_DRAFT.md"))
    parser.add_argument("--output", type=Path, default=Path("paper/MANUSCRIPT_DRAFT.docx"))
    args = parser.parse_args()
    export(args.source, args.output)
    print(f"Exported {args.output} from {args.source}")


if __name__ == "__main__":
    main()
