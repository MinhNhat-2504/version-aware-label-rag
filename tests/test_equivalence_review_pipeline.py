"""Frozen passage candidates require two readers and a third manual decision."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import uuid4

from labelqa.check_equivalence_reviews import check, frozen_candidates
from labelqa.compile_equivalence_reviews import compile_final, main as compile_main
from labelqa.equivalence import load_adjudications


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class EquivalenceReviewPipelineTests(unittest.TestCase):
    def artifact(self, root: Path, name: str) -> Path:
        path = root.with_name(root.name + "_" + name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def fixture(self, root: Path):
        source = self.artifact(root, "source.jsonl")
        item = {"qid": "q1", "set_id": "label-1", "question": "What is stated?",
                "gold_index": 3, "gold_text": "Original source fact.",
                "near_candidates_unverified": [
                    {"index": 5, "text": "Candidate wording of the fact."},
                    {"index": 7, "text": "Unrelated statement."}]}
        write_jsonl(source, [item])
        manifest = self.artifact(root, "equivalence_validation_manifest.json")
        manifest.write_text(json.dumps({"status":
            "development_validation_candidates_pending_two_independent_readers",
            "source": str(source), "source_sha256": digest(source),
            "candidate_count": 2, "question_count": 1}), encoding="utf-8")
        blanks = []
        for candidate in item["near_candidates_unverified"]:
            blanks.append({"review_id": f"q1:{candidate['index']}", "qid": "q1",
                           "set_id": "label-1", "question": item["question"],
                           "original_index": 3, "original_passage": item["gold_text"],
                           "candidate_index": candidate["index"],
                           "candidate_passage": candidate["text"],
                           "reviewer_id": "", "decision": "",
                           "supported_atomic_fact": "", "rationale": ""})
        blank_a = self.artifact(root, "blank_a.jsonl")
        blank_b = self.artifact(root, "blank_b.jsonl")
        write_jsonl(blank_a, blanks)
        write_jsonl(blank_b, blanks)
        return manifest, blank_a, blank_b, blanks

    def test_frozen_identity_manual_compilation_and_receipt(self):
        root = Path("results") / (".test_equivalence_" + uuid4().hex)
        self.assertTrue(root.resolve().is_relative_to(Path.cwd().resolve()))
        with self.subTest("synthetic equivalence pipeline"):
            manifest, blank_a, blank_b, candidates = self.fixture(root)
            blank = frozen_candidates(manifest, blank_a, blank_b)
            reader_a = {row["review_id"]: {**row, "reviewer_id": "reader-a",
                        "decision": "equivalent", "supported_atomic_fact": "Source fact",
                        "rationale": "Compared source wording."} for row in candidates}
            reader_b = deepcopy(reader_a)
            for row in reader_b.values():
                row["reviewer_id"] = "reader-b"
            reader_b["q1:7"]["decision"] = "uncertain"
            queue = check(blank, reader_a, reader_b)
            self.assertTrue(queue["all_items_require_manual_adjudication"])
            self.assertEqual(len(queue["queue"]), 2)
            self.assertEqual(queue["decision_agreement"]["exact_count"], 1)
            self.assertAlmostEqual(queue["decision_agreement"]["cohen_kappa"], 0)
            changed = deepcopy(reader_b)
            changed["q1:7"]["candidate_passage"] = "Tampered source."
            with self.assertRaisesRegex(ValueError, "Frozen candidate"):
                check(blank, reader_a, changed)
            invalid_id = deepcopy(reader_b)
            invalid_id["q1:7"]["reviewer_id"] = None
            with self.assertRaisesRegex(ValueError, "Reviewer IDs must be text"):
                check(blank, reader_a, invalid_id)
            with self.assertRaisesRegex(ValueError, "resolution mode"):
                compile_final(queue, queue, blank)
            edited = deepcopy(queue)
            for index, row in enumerate(edited["queue"]):
                row.update({"adjudicator_id": "expert-c", "adjudication_notes": "Checked source.",
                            "final_decision": "equivalent" if index == 0 else "uncertain"})
            final = compile_final(queue, edited, blank)
            self.assertEqual([row["decision"] for row in final], ["accept", "uncertain"])
            consensus = deepcopy(edited)
            for row in consensus["queue"]:
                row.update({"resolution_mode": "two_reader_consensus",
                            "consensus_reviewer_ids": ["reader-a", "reader-b"],
                            "adjudicator_id": ""})
            consensus_final = compile_final(queue, consensus, blank)
            self.assertTrue(all(row["resolver_ids"] == ["reader-a", "reader-b"]
                                for row in consensus_final))
            filled_a = self.artifact(root, "filled_a.jsonl")
            filled_b = self.artifact(root, "filled_b.jsonl")
            write_jsonl(filled_a, list(reader_a.values()))
            write_jsonl(filled_b, list(reader_b.values()))
            reviewed_queue = self.artifact(root, "queue.json")
            consensus["source_sha256"] = {str(path): digest(path) for path in
                                          (manifest, blank_a, blank_b, filled_a, filled_b)}
            reviewed_queue.write_text(json.dumps(consensus), encoding="utf-8")
            final_path = self.artifact(root, "final.jsonl")
            self.addCleanup(lambda: final_path.with_suffix(".manifest.json").unlink(missing_ok=True))
            with patch("sys.argv", ["compile_equivalence_reviews", "--manifest", str(manifest),
                                    "--blank-a", str(blank_a), "--blank-b", str(blank_b),
                                    "--reader-a", str(filled_a), "--reader-b", str(filled_b),
                                    "--adjudicated-queue", str(reviewed_queue),
                                    "--output", str(final_path)]):
                compile_main()
            decisions, status, counts = load_adjudications(final_path)
            self.assertEqual(status,
                             "independently_reviewed_two_reader_consensus_development_validation_subset")
            self.assertEqual(counts["uncertain"], 1)
            self.assertEqual(len(decisions), 2)
            source = Path(json.loads(manifest.read_text(encoding="utf-8"))["source"])
            source.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Frozen equivalence source changed"):
                load_adjudications(final_path)


if __name__ == "__main__":
    unittest.main()
