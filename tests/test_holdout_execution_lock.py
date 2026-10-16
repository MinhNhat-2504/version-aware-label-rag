import json
from pathlib import Path
import unittest
from unittest.mock import patch

from labelqa.lock_holdout_execution import lock_payload, require_fresh_run_paths, verify_lock
import labelqa.lock_holdout_execution as lock_module
from labelqa.run_local import context_window_tight, saved_rows, validate_locked_run


class HoldoutExecutionLockTests(unittest.TestCase):
    def test_lock_detects_changed_analysis_plan_before_prompt_generation(self):
        plan = Path("README.md")
        gold_path = Path("data/holdout/synthetic_manual_gold.jsonl")
        lock_path = Path("README.md")  # Existing path; only its read_text is mocked.
        gold = [{"qid": "synthetic-old", "set_id": "synthetic-pair", "spl_version": 1,
                 "xml_sha256": "synthetic-xml-digest",
                 "source_snapshot": "data/snapshots/version_holdout_sources.jsonl"}]
        receipt = {"gold_sha256": "synthetic-gold-digest"}
        original_digest = lock_module.digest
        changed = {"plan": False}

        def synthetic_digest(path):
            if path.name == "synthetic-pair_v1.xml":
                return "synthetic-xml-digest"
            if path == gold_path.with_suffix(".manifest.json"):
                return "synthetic-gold-manifest-digest"
            if path == plan and changed["plan"]:
                return "tampered-plan-digest"
            if not path.exists():
                return "synthetic-missing-input-digest"
            return original_digest(path)

        with (patch("labelqa.lock_holdout_execution.digest", side_effect=synthetic_digest),
              patch("labelqa.lock_holdout_execution.model_weights",
                    return_value=([0.1], "synthetic-selector-digest"))):
            frozen = lock_payload(gold, gold_path, receipt, plan, Path("results/xml"),
                                  "synthetic-model", "synthetic-model-digest")
            frozen["created_utc"] = "2026-10-16T00:00:00+00:00"
            with patch.object(Path, "read_text", return_value=json.dumps(frozen)):
                verified, _ = verify_lock(lock_path, gold, gold_path, receipt)
                self.assertEqual(verified["answer_model_digest"], "synthetic-model-digest")
                changed["plan"] = True
                with self.assertRaisesRegex(ValueError, "inputs changed after lock"):
                    verify_lock(lock_path, gold, gold_path, receipt)

    def test_lock_rejects_xml_different_from_manual_gold(self):
        gold = [{"qid": "q", "set_id": "pair", "spl_version": 1,
                 "xml_sha256": "expected", "source_snapshot": "version_holdout_sources.jsonl"}]
        with (patch("labelqa.lock_holdout_execution.digest", return_value="changed"),
              patch("labelqa.lock_holdout_execution.model_weights", return_value=([], "model"))):
            with self.assertRaisesRegex(ValueError, "Pinned XML differs"):
                lock_payload(gold, Path("gold.jsonl"), {"gold_sha256": "gold"},
                             Path("README.md"), Path("results/xml"),
                             "model", "digest")

    def test_runner_rejects_wrong_seed_or_existing_holdout_output(self):
        locked = {"answer_model": "model", "decoding_options": {"seed": 20261016},
                  "request_timeout_seconds": 180,
                  "prompt_paths": {"bm25": "results/holdout_bm25_prompts.jsonl"},
                  "response_paths": {"bm25": "results/holdout_bm25_responses.jsonl"}}
        prompt = Path(locked["prompt_paths"]["bm25"])
        response = Path(locked["response_paths"]["bm25"])
        validate_locked_run("model", 20261016, 180, prompt, response, "bm25", locked)
        with self.assertRaisesRegex(ValueError, "execution lock"):
            validate_locked_run("model", 20261015, 180, prompt, response, "bm25", locked)
        with patch.object(Path, "exists", return_value=True):
            with self.assertRaisesRegex(ValueError, "execution lock"):
                validate_locked_run("model", 20261016, 180, prompt, response, "bm25", locked)

    def test_resume_is_explicit_and_never_repeats_a_saved_call(self):
        locked = {"answer_model": "model", "decoding_options": {"seed": 20261016},
                  "request_timeout_seconds": 180,
                  "prompt_paths": {"bm25": "results/holdout_bm25_prompts.jsonl"},
                  "response_paths": {"bm25": "results/holdout_bm25_responses.jsonl"}}
        prompt = Path(locked["prompt_paths"]["bm25"])
        response = Path(locked["response_paths"]["bm25"])
        with patch.object(Path, "exists", return_value=True):
            validate_locked_run("model", 20261016, 180, prompt, response, "bm25", locked, resume=True)
        saved = "\n".join(json.dumps(row) for row in (
            {"qid": "q1", "parsed_response": {}}, {"qid": "q2", "error": "timed out"}))
        with (patch.object(Path, "exists", return_value=True),
              patch.object(Path, "read_text", return_value=saved)):
            # A locked run keeps the saved error; an unlocked pilot retries it.
            self.assertEqual(set(saved_rows(response, True, {"q1": "h", "q2": "h", "q3": "h"})), {"q1", "q2"})
            self.assertEqual(set(saved_rows(response, False, {})), {"q1"})
            with self.assertRaisesRegex(ValueError, "unknown or repeated"):
                saved_rows(response, True, {"q1": "h"})

    def test_context_window_flag(self):
        options = {"num_ctx": 8192, "num_predict": 512}
        self.assertFalse(context_window_tight(7000, options))
        self.assertTrue(context_window_tight(7900, options))
        self.assertFalse(context_window_tight(None, options))

    def test_lock_requires_fresh_declared_run_paths(self):
        fresh = {"prompt_paths": {"bm25": "results/.never_created_lock_prompt.jsonl"},
                 "response_paths": {"bm25": "results/.never_created_lock_response.jsonl"}}
        require_fresh_run_paths(fresh)
        occupied = {"prompt_paths": {"bm25": "README.md"},
                    "response_paths": {"bm25": "results/.never_created_lock_response.jsonl"}}
        with self.assertRaisesRegex(ValueError, "requires fresh"):
            require_fresh_run_paths(occupied)

    def test_lock_rejects_non_utc_creation_timestamp(self):
        plan = Path("README.md")
        gold_path = Path("data/holdout/synthetic_manual_gold.jsonl")
        gold = [{"qid": "synthetic-old", "set_id": "synthetic-pair", "spl_version": 1,
                 "xml_sha256": "synthetic-xml-digest",
                 "source_snapshot": "data/snapshots/version_holdout_sources.jsonl"}]
        receipt = {"gold_sha256": "synthetic-gold-digest"}
        original_digest = lock_module.digest

        def synthetic_digest(path):
            if path.name == "synthetic-pair_v1.xml":
                return "synthetic-xml-digest"
            if path == gold_path.with_suffix(".manifest.json"):
                return "synthetic-gold-manifest-digest"
            if not path.exists():
                return "synthetic-missing-input-digest"
            return original_digest(path)

        with (patch("labelqa.lock_holdout_execution.digest", side_effect=synthetic_digest),
              patch("labelqa.lock_holdout_execution.model_weights",
                    return_value=([0.1], "synthetic-selector-digest"))):
            frozen = lock_payload(gold, gold_path, receipt, plan, Path("results/xml"),
                                  "synthetic-model", "synthetic-model-digest")
            frozen["created_utc"] = "2026-10-16T07:00:00"
            with patch.object(Path, "read_text", return_value=json.dumps(frozen)):
                with self.assertRaisesRegex(ValueError, "must be UTC"):
                    verify_lock(plan, gold, gold_path, receipt)


if __name__ == "__main__":
    unittest.main()
