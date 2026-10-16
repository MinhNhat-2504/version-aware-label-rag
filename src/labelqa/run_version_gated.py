"""Run a version-scoped pilot with the absence gate before local model calls."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl
from .run_local import model_digest, post_json
from .version_gate import absent_from_label, target_entity


OPTIONS = {"temperature": 0, "num_ctx": 8192, "num_predict": 512}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/snapshots/version_pair_paxlovid.jsonl"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; refusing to mix a new run with saved responses")
    pair = next(read_jsonl(args.source))
    versions = {v["spl_version"]: v for v in pair["versions"]}
    prompts = list(read_jsonl(args.prompts))
    digest = model_digest(args.model)
    options = {**OPTIONS, **({"seed": args.seed} if args.seed is not None else {})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"gate_outputs": 0, "model_calls": 0, "errors": 0}
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            version = prompt["spl_version"]
            source = versions[version]
            if prompt["set_id"] != pair["set_id"] or prompt["xml_sha256"] != source["xml_sha256"]:
                raise ValueError(f"Pinned source mismatch for {prompt['qid']}")
            prompt_hash = sha256((prompt["system_prompt"] + prompt["user_prompt"]).encode()).hexdigest()
            entity = target_entity(prompt["question"])
            row = {"qid": prompt["qid"], "task": prompt["task"], "model": args.model,
                   "model_digest": digest, "prompt_sha256": prompt_hash,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "decoding_options": options, "gate_target_entity": entity}
            if entity and absent_from_label(entity, source["sections"]):
                parsed = {"answer": f"The parsed sections of this pinned SPL version do not mention {entity}; the interaction cannot be determined from these sections.",
                          "claims": [], "abstain": True}
                row.update({"raw_response": json.dumps(parsed), "parsed_response": parsed,
                            "answer_source": "deterministic_version_gate", "model_called": False,
                            "prompt_eval_count": 0, "eval_count": 0, "total_duration_ns": 0})
                counts["gate_outputs"] += 1
            else:
                row.update({"answer_source": "model_response", "model_called": True})
                counts["model_calls"] += 1
                try:
                    response = post_json("/api/chat", {
                        "model": args.model,
                        "messages": [{"role": "system", "content": prompt["system_prompt"]},
                                     {"role": "user", "content": prompt["user_prompt"]}],
                        "stream": False, "format": "json", "options": options,
                    }, args.timeout)
                    raw = response.get("message", {}).get("content", "")
                    row.update({"raw_response": raw, "parsed_response": json.loads(raw),
                                "prompt_eval_count": response.get("prompt_eval_count"),
                                "eval_count": response.get("eval_count"),
                                "total_duration_ns": response.get("total_duration")})
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    row["error"] = str(exc)
                    counts["errors"] += 1
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"qid": row["qid"], "model_called": row["model_called"]}), flush=True)
    print(json.dumps({"output": str(args.output), **counts}))


if __name__ == "__main__":
    main()
