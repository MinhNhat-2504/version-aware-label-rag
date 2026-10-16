"""Merged-cell coordinates are a structural warning, not clinical annotation."""

import unittest
from xml.etree import ElementTree as ET

from labelqa.audit_spl_tables import table_layout_rows


NS = "urn:hl7-org:v3"


class TableLayoutTests(unittest.TestCase):
    def test_rowspan_moves_following_cell_to_its_actual_column(self):
        table = ET.fromstring(
            f'<table xmlns="{NS}"><tbody>'
            '<tr><td rowspan="2">Drug A</td><td>Day 1</td><td>Dose 1</td></tr>'
            '<tr><td>Day 2</td><td>Dose 2</td></tr>'
            '</tbody></table>'
        )
        rows = table_layout_rows(table)
        self.assertEqual(rows[1]["carried_columns"], [0])
        self.assertEqual([c["column"] for c in rows[1]["cells"]], [1, 2])
        self.assertEqual([c["ordinal"] for c in rows[1]["cells"]], [0, 1])

    def test_colspan_and_invalid_span(self):
        table = ET.fromstring(
            f'<table xmlns="{NS}"><tbody>'
            '<tr><th colspan="2">Dose</th><th>Effect</th></tr>'
            '</tbody></table>'
        )
        self.assertEqual([c["column"] for c in table_layout_rows(table)[0]["cells"]], [0, 2])
        table[0][0][0].set("colspan", "0")
        with self.assertRaises(ValueError):
            table_layout_rows(table)

    def test_nested_table_fails_closed(self):
        table = ET.fromstring(
            f'<table xmlns="{NS}"><tr><td><table><tr><td>Nested</td></tr>'
            '</table></td></tr></table>'
        )
        with self.assertRaisesRegex(ValueError, "Nested SPL tables"):
            table_layout_rows(table)


if __name__ == "__main__":
    unittest.main()
