"""Run frozen prompt JSONL against an installed local Ollama model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from .data import portable_path

BASE = "http://127.0.0.1:11434"
OPENER = build_opener(ProxyHandler({}))


def post_json(path: str, payload: dict, timeout: int) -> dict:
    request = Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(request, timeout=timeout) as response:
        return json.load(response)


def model_digest(model: str) -> str:
    with OPENER.open(BASE + "/api/tags", timeout=30) as response:
        models = json.load(response).get("models", [])
    matches = [item for item in models if item.get("name") == model or item.get("model") == model]
    if not matches:
        raise ValueError(f"Model {model!r} is not installed in local Ollama")
    return matches[0]["digest"]


def validate_locked_run(model: str, seed: int | None, timeout: int,
                        input_path: Path, output_path: Path, method: str, locked: dict) -> None:
    if (model != locked["answer_model"] or seed != locked["decoding_options"]["seed"]
            or timeout != locked["request_timeout_seconds"]
            or input_path != portable_path(locked["prompt_paths"][method])
            or output_path != portable_path(locked["response_paths"][method])
            or output_path.exists()):
        raise ValueError("Holdout model, seed, timeout, paths or existing output differ from execution lock")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--seed", type=int, default=None,
                        help="Explicit Ollama generation seed; omitted for legacy pilot parity")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--gold", type=Path, default=Path("data/holdout/version_gold_adjudicated.jsonl"))
    parser.add_argument("--execution-lock", type=Path, default=Path("paper/HOLDOUT_EXECUTION_LOCK.json"))
    args = parser.parse_args()
    holdout_receipt = args.input.with_suffix(".manifest.json")
    locked = None
    if holdout_receipt.is_file():
        receipt = json.loads(holdout_receipt.read_text(encoding="utf-8"))
        if receipt.get("status") == "holdout_prompts_after_manual_gold":
            from .lock_holdout_execution import verify_lock
            from .prepare_holdout_eval import load_manual_gold, model_weights
            from .summarize_holdout_retrieval import METHODS, read_prompts
            gold, gold_receipt = load_manual_gold(args.gold)
            locked, lock_hash = verify_lock(args.execution_lock, gold, args.gold, gold_receipt)
            method = receipt.get("method")
            if method not in METHODS:
                raise ValueError("Unknown holdout method in prompt receipt")
            validate_locked_run(args.model, args.seed, args.timeout, args.input, args.output,
                                method, locked)
            expected_selector = model_weights()[1] if method == "learned_cover" else None
            gold_manifest_hash = sha256(args.gold.with_suffix(".manifest.json").read_bytes()).hexdigest()
            prompts = read_prompts(args.input, method, gold_receipt["gold_sha256"],
                                   gold_manifest_hash, expected_selector, lock_hash)
            if set(prompts) != {row["qid"] for row in gold}:
                raise ValueError("Holdout prompt questions differ from adjudicated gold")
            from .prepare_blinded_answer_review import load_responses
            for prior in METHODS[:METHODS.index(method)]:
                prior_prompt = portable_path(locked["prompt_paths"][prior])
                prior_response = portable_path(locked["response_paths"][prior])
                if not prior_response.is_file():
                    raise ValueError(f"Earlier holdout method is incomplete: {prior}")
                prior_prompts = read_prompts(prior_prompt, prior, gold_receipt["gold_sha256"],
                                             gold_manifest_hash,
                                             model_weights()[1] if prior == "learned_cover" else None,
                                             lock_hash)
                previous = load_responses(prior_response, prior_prompts)
                if any(row.get("model") != locked["answer_model"] or
                       row.get("model_digest") != locked["answer_model_digest"] or
                       row.get("execution_lock_sha256") != lock_hash or
                       row.get("decoding_options") != locked["decoding_options"]
                       for row in previous.values()):
                    raise ValueError(f"Earlier holdout method differs from execution lock: {prior}")
    digest = model_digest(args.model)
    if locked is not None and digest != locked["answer_model_digest"]:
        raise ValueError("Installed answering model digest differs from execution lock")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    finished = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                previous = json.loads(line)
                if not previous.get("error"):
                    finished[previous["qid"]] = previous
    completed = errors = 0
    with args.input.open("r", encoding="utf-8") as source, args.output.open("a", encoding="utf-8") as sink:
        for line in source:
            item = json.loads(line)
            prompt_hash = sha256((item["system_prompt"] + item["user_prompt"]).encode()).hexdigest()
            options = {"temperature": 0, "num_ctx": 8192, "num_predict": 512,
                       **({"seed": args.seed} if args.seed is not None else {})}
            if item["qid"] in finished:
                previous = finished[item["qid"]]
                if (previous.get("prompt_sha256") != prompt_hash
                        or previous.get("model_digest") != digest
                        or previous.get("decoding_options") != options):
                    raise ValueError(f"Saved response for {item['qid']} uses another prompt, model, or decoding options")
                continue
            row = {
                "qid": item["qid"],
                "task": item["task"],
                "model": args.model,
                "model_digest": digest,
                "prompt_sha256": prompt_hash,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "decoding_options": options,
            }
            if locked is not None:
                row["execution_lock_sha256"] = lock_hash
            try:
                response = post_json("/api/chat", {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["user_prompt"]},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": row["decoding_options"],
                }, args.timeout)
                raw = response.get("message", {}).get("content", "")
                row["raw_response"] = raw
                row["parsed_response"] = json.loads(raw)
                row["prompt_eval_count"] = response.get("prompt_eval_count")
                row["eval_count"] = response.get("eval_count")
                row["total_duration_ns"] = response.get("total_duration")
                completed += 1
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                row["error"] = str(exc)
                errors += 1
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            print(json.dumps({"qid": item["qid"], "completed": completed, "errors": errors}), flush=True)
    print(json.dumps({"output": str(args.output), "completed": completed, "errors": errors}))


if __name__ == "__main__":
    main()
