import unittest

from labelqa.spl_tables import parse_structured_passages


NESTED = b"""<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="urn:hl7-org:v3"><component><structuredBody>
  <component><section><title>PATIENT INFORMATION</title><text>
    <table><tbody>
      <tr><td>Outer intro text.</td></tr>
      <tr><td>Before the list.
        <table><thead><tr><th>Drug</th><th>Advice</th></tr></thead>
          <tbody><tr><td>alpha</td><td>Avoid alpha.</td></tr>
                 <tr><td>beta</td><td>Reduce beta.</td></tr></tbody></table>
      </td></tr>
    </tbody></table>
  </text></section></component>
</structuredBody></component></document>"""


class NestedTableTests(unittest.TestCase):
    def test_nested_table_rows_belong_only_to_the_inner_table(self):
        rows = [p for p in parse_structured_passages(NESTED) if p.kind == "table_row"]
        outer = [p for p in rows if p.table_index == 0]
        inner = [p for p in rows if p.table_index == 1]
        self.assertEqual([p.row_index for p in outer], [0, 1])
        self.assertIn("Before the list.", outer[1].text)
        self.assertNotIn("Avoid alpha", outer[1].text)
        self.assertEqual([p.row_index for p in inner], [1, 2])
        self.assertIn("Drug: alpha | Advice: Avoid alpha.", inner[0].text)
        self.assertIn("Drug: beta | Advice: Reduce beta.", inner[1].text)


if __name__ == "__main__":
    unittest.main()
