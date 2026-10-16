from io import BytesIO
import unittest
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from labelqa.export_manuscript_docx import W, build_docx_bytes


class ExportManuscriptDocxTests(unittest.TestCase):
    def test_editable_docx_preserves_heading_table_and_link(self):
        sample = """# Paper title

**Working draft.** One *important* result links to [source](https://example.org/paper).

## 1. Results

| Method | Result |
| --- | ---: |
| BM25 | 2/3 |
"""
        with ZipFile(BytesIO(build_docx_bytes(sample))) as archive:
            self.assertIsNone(archive.testzip())
            document = ET.fromstring(archive.read("word/document.xml"))
            rels = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
            types = ET.fromstring(archive.read("[Content_Types].xml"))
            self.assertEqual(len(document.findall(f".//{{{W}}}tbl")), 1)
            self.assertEqual(len(document.findall(f".//{{{W}}}hyperlink")), 1)
            self.assertIn("Paper title", "".join(document.itertext()))
            self.assertIn("BM25", "".join(document.itertext()))
            self.assertTrue(any(rel.get("Target") == "https://example.org/paper"
                                for rel in rels))
            self.assertTrue(any(item.get("PartName") == "/word/document.xml"
                                for item in types))


if __name__ == "__main__":
    unittest.main()
