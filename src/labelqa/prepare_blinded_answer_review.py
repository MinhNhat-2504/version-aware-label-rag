"""Prepare method-blind answer review packets only after manual holdout gold."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import secrets

from .data import portable_path
from .prepare_holdout_eval import DEFAULT_GOLD, load_manual_gold, model_weights
from .score_qa import validate_provenance
from .summarize_holdout_retrieval import METHODS, read_prompts


def load_responses(path: Path, prompts: dict[str, dict]) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len({row["qid"] for row in rows}) or set(prompts) != {row["qid"] for row in rows}:
        raise ValueError(f"Responses must contain exactly one row per accepted question: {path}")
    responses = {row["qid"]: row for row in rows}
    validate_provenance(prompts, responses)
    return responses


def review_packet(gold: list[dict], prompts: dict[str, dict[str, dict]],
                  responses: dict[str, dict[str, dict]], id_factory=None,
                  shuffle_seed: int = 20261016) -> tuple[list[dict], list[dict]]:
    """Separate reviewer-visible outputs from private method/provenance mapping."""
    id_factory = id_factory or (lambda: secrets.token_hex(16))
    qids = {row["qid"] for row in gold}
    if any(set(prompts[method]) != qids or set(responses[method]) != qids for method in METHODS):
        raise ValueError("Method QIDs differ from accepted manual gold")
    configurations = {(row.get("model_digest"), json.dumps(row.get("decoding_options"), sort_keys=True))
                      for method in METHODS for row in responses[method].values()}
    if (len(configurations) != 1 or not all(isinstance(row.get("model_digest"), str)
            and row["model_digest"] and isinstance(row.get("decoding_options"), dict)
            for method in METHODS for row in responses[method].values())):
        raise ValueError("Answering model or decoding options differ across methods")
    packet, private_map = [], []
    for gold_row in gold:
        qid = gold_row["qid"]
        for method in METHODS:
            prompt, response = prompts[method][qid], responses[method][qid]
            for key in ("qid", "set_id", "spl_version", "question", "xml_sha256"):
                if prompt.get(key) != gold_row.get(key):
                    raise ValueError(f"Gold/prompt identity differs: {qid}, {method}, {key}")
            marker = "Label passages:\n"
            if marker not in prompt["user_prompt"]:
                raise ValueError(f"Visible evidence missing from prompt: {qid}, {method}")
            review_id = id_factory()
            if not isinstance(review_id, str) or len(review_id) < 12:
                raise ValueError("Review ID must be an opaque string of at least 12 characters")
            parsed = response.get("parsed_response")
            packet.append({
                "review_id": review_id, "qid": qid, "set_id": gold_row["set_id"],
                "spl_version": gold_row["spl_version"], "xml_sha256": gold_row["xml_sha256"],
                "source_snapshot": gold_row["source_snapshot"],
                "question": gold_row["question"],
                "visible_evidence": prompt["user_prompt"].split(marker, 1)[1],
                "model_answer": parsed if isinstance(parsed, dict) else None,
                "raw_model_output": response.get("raw_response", ""),
                "generation_error": bool(response.get("error")),
                "reviewer_id": "", "review_status": "", "answer_complete": None,
                "all_claims_supported": None,
                "citations_at_correct_location": None, "correct_version": None,
                "refusal_appropriate": None, "unsafe_unsupported_claim": None,
                "claim_judgments": [], "review_notes": "",
            })
            private_map.append({"review_id": review_id, "qid": qid, "method": method,
                                "prompt_sha256": response["prompt_sha256"],
                                "raw_response_sha256": sha256(response.get("raw_response", "").encode()).hexdigest(),
                                "model_digest": response.get("model_digest")})
    if len({row["review_id"] for row in packet}) != len(packet):
        raise ValueError("Duplicate blind review ID")
    order = list(range(len(packet)))
    random.Random(shuffle_seed).shuffle(order)
    return [packet[i] for i in order], [private_map[i] for i in order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--execution-lock", type=Path, default=Path("paper/HOLDOUT_EXECUTION_LOCK.json"))
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}-prompts", type=Path,
                            default=Path(f"results/holdout_{method}_prompts.jsonl"))
        parser.add_argument(f"--{method.replace('_', '-')}-responses", type=Path,
                            default=Path(f"results/holdout_{method}_responses.jsonl"))
    parser.add_argument("--output", type=Path, required=True,
                        help="Method-blind packet to copy separately for each reader")
    parser.add_argument("--private-map", type=Path, required=True,
                        help="Method key; store apart from the readers")
    args = parser.parse_args()
    if args.output.resolve() == args.private_map.resolve() or args.output.exists() or args.private_map.exists():
        parser.error("Use two distinct, new output paths")
    gold, receipt = load_manual_gold(args.gold)  # Gate before prompt/response access.
    from .lock_holdout_execution import verify_lock
    lock, lock_hash = verify_lock(args.execution_lock, gold, args.gold, receipt)
    gold_manifest_hash = sha256(args.gold.with_suffix(".manifest.json").read_bytes()).hexdigest()
    selector_hash = model_weights()[1]
    prompt_paths = {method: getattr(args, method + "_prompts") for method in METHODS}
    response_paths = {method: getattr(args, method + "_responses") for method in METHODS}
    if any(prompt_paths[method] != portable_path(lock["prompt_paths"][method]) or
           response_paths[method] != portable_path(lock["response_paths"][method])
           for method in METHODS):
        raise ValueError("Answer-review input paths differ from execution lock")
    prompts = {method: read_prompts(prompt_paths[method], method, receipt["gold_sha256"],
                                    gold_manifest_hash,
                                    selector_hash if method == "learned_cover" else None,
                                    lock_hash)
               for method in METHODS}
    responses = {method: load_responses(response_paths[method], prompts[method])
                 for method in METHODS}
    for method in METHODS:
        if any(row.get("model") != lock["answer_model"] or
               row.get("model_digest") != lock["answer_model_digest"] or
               row.get("execution_lock_sha256") != lock_hash or
               row.get("decoding_options") != lock["decoding_options"]
               for row in responses[method].values()):
            raise ValueError(f"Answering model/settings differ from execution lock: {method}")
    packet, private_map = review_packet(gold, prompts, responses)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.private_map.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packet),
                           encoding="utf-8")
    key = {"status": "private_method_key_keep_from_reviewers",
           "gold_sha256": receipt["gold_sha256"], "execution_lock_sha256": lock_hash,
           "packet_sha256": sha256(args.output.read_bytes()).hexdigest(),
           "prompt_sha256": {m: sha256(p.read_bytes()).hexdigest() for m, p in prompt_paths.items()},
           "response_sha256": {m: sha256(p.read_bytes()).hexdigest() for m, p in response_paths.items()},
           "mapping": private_map}
    args.private_map.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"packet": str(args.output), "private_map": str(args.private_map),
                      "answers": len(packet), "accepted_questions": len(gold)}))


if __name__ == "__main__":
    main()
