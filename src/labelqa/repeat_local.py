"""Measure exact-prompt output variability on development questions only."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .run_local import model_digest, post_json
from .score_qa import score_one


OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 512}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--task", choices=("factual", "multihop", "refusal"), default="factual")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.output.exists():
        parser.error("Require at least two repeats and a new output file")
    prompts = [p for p in read_jsonl(args.prompts) if p["task"] == args.task]
    if not prompts:
        parser.error("No prompts with selected task")
    digest = model_digest(args.model)
    options = {**OPTIONS, **({"seed": args.seed} if args.seed is not None else {})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = Counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for repeat in range(1, args.repeats + 1):
            for prompt in prompts:
                prompt_hash = sha256((prompt["system_prompt"] + prompt["user_prompt"]).encode()).hexdigest()
                row = {"qid": prompt["qid"], "repeat": repeat, "model": args.model,
                       "model_digest": digest, "prompt_sha256": prompt_hash,
                       "decoding_options": options,
                       "created_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    response = post_json("/api/chat", {
                        "model": args.model,
                        "messages": [{"role": "system", "content": prompt["system_prompt"]},
                                     {"role": "user", "content": prompt["user_prompt"]}],
                        "stream": False, "format": "json", "options": options,
                    }, 180)
                    raw = response.get("message", {}).get("content", "")
                    row["raw_response"] = raw
                    row["parsed_response"] = json.loads(raw)
                    row["total_duration_ns"] = response.get("total_duration")
                    metrics = score_one(prompt, row)
                    row["valid_schema"] = metrics.get("valid_schema")
                    row["correct_refusal_decision"] = metrics.get("correct_refusal_decision")
                    parsed = row["parsed_response"]
                    summary[(prompt["qid"], "abstain",
                             parsed.get("abstain") if isinstance(parsed, dict) else None)] += 1
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    row["error"] = str(exc)
                    summary[(prompt["qid"], "error")] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(json.dumps({"qid": row["qid"], "repeat": repeat,
                                  "abstain": (row.get("parsed_response") or {}).get("abstain"),
                                  "error": row.get("error")}), flush=True)
    print(json.dumps({"output": str(args.output), "summary": [
        {"key": list(key), "count": count} for key, count in summary.items()]}))


if __name__ == "__main__":
    main()
