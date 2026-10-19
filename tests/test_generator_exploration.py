import unittest

from labelqa.summarize_generator_exploration import empty_answer, repeatability, size_contrast
from labelqa.summarize_holdout_retrieval import METHODS


def run(name, model, outcomes, answers):
    per_item = [{"qid": f"q{i}", "set_id": f"pair{i // 2}", **{m: outcome for m in METHODS}}
                for i, outcome in enumerate(outcomes)]
    responses = {m: {f"q{i}": {"qid": f"q{i}", "parsed_response": {"answer": answer}}
                     for i, answer in enumerate(answers)} for m in METHODS}
    counts = {m: {"mechanical_success": sum(outcomes)} for m in METHODS}
    return {"name": name, "model": model, "responses": responses,
            "report": {"per_item": per_item, "method_counts": counts}}


class GeneratorExplorationTests(unittest.TestCase):
    def test_repeatability_counts_items_with_identical_outcomes_and_text(self):
        first = run("a", "m", [True, False, True, True], ["x", "", "y", "z"])
        second = run("b", "m", [True, True, True, True], ["x", "w", "y", "changed"])
        result = repeatability([first, second])["bm25"]
        self.assertEqual(result["identical_pass_fail_all_runs"], 3)
        self.assertEqual(result["identical_parsed_response_all_runs"], 2)
        self.assertEqual(result["mechanical_success_by_run"], [3, 4])

    def test_size_contrast_pairs_items_by_question(self):
        small = run("small", "3b", [False, False, True, True], ["", "", "a", "b"])
        large = run("large", "7b", [True, True, True, True], ["a", "b", "c", "d"])
        contrast = size_contrast(small, large)["mechanical_success"]["learned_cover"]
        self.assertEqual(contrast["paired_cells"], {"both": 2, "left_only": 2, "right_only": 0, "neither": 0})
        self.assertEqual(contrast["difference_percentage_points"], 50.0)

    def test_empty_answer_ignores_generation_errors(self):
        self.assertTrue(empty_answer({"parsed_response": {"answer": " "}}))
        self.assertFalse(empty_answer({"parsed_response": None, "error": "timeout"}))


if __name__ == "__main__":
    unittest.main()
