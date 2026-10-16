import unittest

from labelqa.check_holdout_reviews import validate_span_locators
from labelqa.spl_tables import parse_structured_passages
from labelqa.synthetic_demo import SYNTHETIC_XML


PASSAGES = parse_structured_passages(SYNTHETIC_XML)


def span(section, kind, quote, table=None, row=None):
    return {"section_index": section, "source_kind": kind, "quote": quote,
            "table_index": table, "row_index": row}


class ReviewLocatorTests(unittest.TestCase):
    def test_exact_prose_and_table_row_locators_pass(self):
        validate_span_locators("q", [[span(0, "prose", "indicated to treat Condition Alpha")],
                                     [span(1, "table_row", "One synthetic unit once daily.", 0, 1)]],
                               PASSAGES)

    def test_wrong_row_or_kind_fails_before_gold(self):
        with self.assertRaisesRegex(ValueError, "table 0, row 0"):
            validate_span_locators("q", [[span(1, "table_row", "One synthetic unit", 0, 0)]], PASSAGES)
        with self.assertRaisesRegex(ValueError, "prose window"):
            validate_span_locators("q", [[span(1, "prose", "One synthetic unit")]], PASSAGES)

    def test_quote_crossing_passages_fails(self):
        with self.assertRaises(ValueError):
            validate_span_locators("q", [[span(0, "prose", "in adults. Store the synthetic package")]],
                                   PASSAGES)


if __name__ == "__main__":
    unittest.main()
