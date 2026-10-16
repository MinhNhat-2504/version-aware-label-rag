import json
import shutil
import unittest
from pathlib import Path
from unittest import mock

from labelqa import benchmark_test
from labelqa.benchmark_test import decision, eligible_test_questions, paired_cells, verify_lock
from labelqa.data import Question, split_label
from labelqa.export_reviewer_interface import ROOT


def label_in(split):
    return next(f"label-{i}" for i in range(10_000) if split_label(f"label-{i}") == split)


class BenchmarkTestPreregistrationTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "results" / ".test-benchmark-lock"
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_decision_requires_interval_to_exclude_zero(self):
        self.assertEqual(decision([0.001, 0.05]), "candidate_superior_on_retrieval_proxy")
        self.assertEqual(decision([0.0, 0.05]), "inconclusive_interval_includes_zero")
        self.assertEqual(decision([-0.04, -0.01]), "candidate_inferior_on_retrieval_proxy")

    def test_only_test_split_multihop_with_scoreable_gold_is_eligible(self):
        test, dev = label_in("test"), label_in("dev")
        questions = [
            Question("a", test, "multihop", "q", (1, 2)),
            Question("b", test, "factual", "q", (1,)),
            Question("c", test, "multihop", "q", tuple(range(6))),
            Question("d", dev, "multihop", "q", (1, 2)),
        ]
        self.assertEqual([q.qid for q in eligible_test_questions(questions)], ["a"])

    def test_paired_cells_follow_candidate_and_baseline(self):
        q = Question("a", "s", "multihop", "q", (1, 2))
        cells = paired_cells([(q, [1, 2]), (q, [1])], [(q, [3]), (q, [1, 2])])
        self.assertEqual(cells, {"both": 0, "candidate_only": 1, "baseline_only": 1, "neither": 0})

    def test_missing_or_changed_lock_blocks_evaluation(self):
        path = self.root / "lock.json"
        with self.assertRaises(ValueError):
            verify_lock(path, [], self.root)
        expected = {"status": "x", "denominator": {"questions": 2}}
        path.write_text(json.dumps({**expected, "created_utc": "2026-10-17T00:00:00+00:00"}),
                        encoding="utf-8")
        with mock.patch.object(benchmark_test, "lock_payload", return_value=expected):
            locked, _ = verify_lock(path, [], self.root)
            self.assertEqual(locked["denominator"]["questions"], 2)
        changed = {**expected, "denominator": {"questions": 3}}
        with mock.patch.object(benchmark_test, "lock_payload", return_value=changed):
            with self.assertRaises(ValueError):
                verify_lock(path, [], self.root)


if __name__ == "__main__":
    unittest.main()
