"""Freeze and verify the post-gold holdout execution inputs before model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import platform
from pathlib import Path

from .data import portable_path
from .prepare_holdout_eval import DEFAULT_GOLD, MODEL_REPORT, load_manual_gold, model_weights
from .prompts import SYSTEM_PROMPT


DEFAULT_LOCK = Path("paper/HOLDOUT_EXECUTION_LOCK.json")
DEFAULT_PLAN = Path("paper/ANALYSIS_PLAN.md")
DEFAULT_XML_DIR = Path("results/pinned_xml_holdout_after_gold")
METHODS = ("bm25", "learned_cover", "ontology_cover")
CODE_FILES = (
    "src/labelqa/answer_review.py", "src/labelqa/check_holdout_reviews.py",
    "src/labelqa/compile_holdout_gold.py", "src/labelqa/data.py",
    "src/labelqa/holdout_reporting.py", "src/labelqa/learned_retrieval.py",
    "src/labelqa/prepare_blinded_answer_review.py", "src/labelqa/prepare_holdout_eval.py",
    "src/labelqa/prompts.py", "src/labelqa/recent.py", "src/labelqa/retrieval.py",
    "src/labelqa/review_resolution.py",
    "src/labelqa/run_local.py",
    "src/labelqa/score_qa.py", "src/labelqa/spl_passages.py", "src/labelqa/spl_tables.py",
    "src/labelqa/summarize_answer_reviews.py", "src/labelqa/summarize_holdout_retrieval.py",
    "src/labelqa/lock_holdout_execution.py",
)
ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def runtime_fingerprint() -> dict[str, str]:
    return {"python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "operating_system": platform.system(),
            "machine": platform.machine()}


def require_fresh_run_paths(payload: dict) -> None:
    """Require unused destinations when the confirmatory execution is locked."""
    prompts = [Path(path) for path in payload["prompt_paths"].values()]
    paths = prompts + [path.with_suffix(".manifest.json") for path in prompts]
    paths += [Path(path) for path in payload["response_paths"].values()]
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise ValueError("Execution lock requires fresh prompt/response paths; already exists: "
                         + ", ".join(existing))


def lock_payload(gold: list[dict], gold_path: Path, receipt: dict, plan: Path,
                 xml_dir: Path, answer_model: str, answer_model_digest: str,
                 seed: int = 20261016, run_prefix: str = "results/holdout") -> dict:
    """Build one exact input fingerprint; no holdout model inference occurs here."""
    if (not gold or not answer_model.strip() or not answer_model_digest.strip()
            or type(seed) is not int or not run_prefix.strip()):
        raise ValueError("Gold, installed answer model digest and fixed seed are required")
    if not plan.is_file():
        raise ValueError("Analysis plan must exist before locking execution")
    selector_hash = model_weights()[1]
    xml_hashes = {}
    for row in gold:
        key = f"{row['set_id']}_v{row['spl_version']}"
        path = xml_dir / f"{key}.xml"
        actual = digest(path)
        if actual != row["xml_sha256"]:
            raise ValueError(f"Pinned XML differs from manual gold: {key}")
        if key in xml_hashes and xml_hashes[key] != actual:
            raise ValueError(f"Conflicting XML identity in gold: {key}")
        xml_hashes[key] = actual
    sources = {}
    for source in sorted({row["source_snapshot"] for row in gold}):
        sources[source] = digest(portable_path(source))
    return {
        "status": "holdout_execution_locked_after_manual_gold_before_model_calls",
        "gold_sha256": receipt["gold_sha256"],
        "gold_manifest_sha256": digest(gold_path.with_suffix(".manifest.json")),
        "analysis_plan": str(plan), "analysis_plan_sha256": digest(plan),
        "source_snapshot_sha256": sources,
        "xml_dir": str(xml_dir), "accepted_xml_sha256": xml_hashes,
        "selector_model_sha256": selector_hash,
        "selector_validation_report_sha256": digest(MODEL_REPORT),
        "code_sha256": {name: digest(ROOT / name) for name in CODE_FILES},
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "methods_in_order": list(METHODS), "passage_budget": 5,
        "run_prefix": run_prefix,
        "prompt_paths": {method: f"{run_prefix}_{method}_prompts.jsonl" for method in METHODS},
        "response_paths": {method: f"{run_prefix}_{method}_responses.jsonl" for method in METHODS},
        "response_schema_version": 2, "calls_per_question_method": 1,
        "request_timeout_seconds": 180,
        "answer_model": answer_model, "answer_model_digest": answer_model_digest,
        "runtime": runtime_fingerprint(),
        "decoding_options": {"temperature": 0, "num_ctx": 8192,
                             "num_predict": 512, "seed": seed},
    }


def verify_lock(path: Path, gold: list[dict], gold_path: Path, receipt: dict) -> tuple[dict, str]:
    if not path.is_file():
        raise ValueError("Frozen execution lock is required before holdout prompts or responses")
    locked = json.loads(path.read_text(encoding="utf-8"))
    if locked.get("status") != "holdout_execution_locked_after_manual_gold_before_model_calls":
        raise ValueError("Unrecognized holdout execution lock")
    options = locked.get("decoding_options")
    if not isinstance(options, dict):
        raise ValueError("Execution lock has no decoding options")
    try:
        created = datetime.fromisoformat(locked["created_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Execution lock has an invalid creation timestamp") from exc
    if created.utcoffset() != timezone.utc.utcoffset(created):
        raise ValueError("Execution lock creation timestamp must be UTC")
    expected = lock_payload(gold, gold_path, receipt, portable_path(locked["analysis_plan"]),
                            portable_path(locked["xml_dir"]), locked["answer_model"],
                            locked["answer_model_digest"], options.get("seed"),
                            locked["run_prefix"])
    if locked != {**expected, "created_utc": locked.get("created_utc")}:
        raise ValueError("Holdout execution inputs changed after lock")
    return locked, digest(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--run-prefix", default="results/holdout")
    parser.add_argument("--output", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite a frozen execution lock")
    gold, receipt = load_manual_gold(args.gold)  # Before XML/model access.
    from .run_local import model_digest
    installed_digest = model_digest(args.model)
    payload = lock_payload(gold, args.gold, receipt, args.plan, args.xml_dir,
                           args.model, installed_digest, args.seed, args.run_prefix)
    require_fresh_run_paths(payload)
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(args.output), "accepted_versions": len(payload["accepted_xml_sha256"]),
                      "gold_sha256": payload["gold_sha256"]}))


if __name__ == "__main__":
    main()
