import unittest

from labelqa.pilot_claim_judge import parse_verdict


class PilotClaimJudgeTests(unittest.TestCase):
    def test_decisive_verdict_requires_quote_in_source(self):
        source = "Take 2 mg once daily."
        result = parse_verdict(
            '{"verdict":"contradicted","quote":"2 mg once daily","reason":"dose mismatch"}',
            source,
        )
        self.assertEqual(result["verdict"], "contradicted")
        with self.assertRaisesRegex(ValueError, "exact source quote"):
            parse_verdict('{"verdict":"supported","quote":"","reason":"yes"}', source)
        with self.assertRaisesRegex(ValueError, "does not occur"):
            parse_verdict(
                '{"verdict":"supported","quote":"4 mg once daily","reason":"yes"}', source
            )

    def test_invalid_schema_fails(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_verdict('[]', "A")
        with self.assertRaisesRegex(ValueError, "schema"):
            parse_verdict('{"verdict":"maybe","quote":"A","reason":"x"}', "A")


if __name__ == "__main__":
    unittest.main()
