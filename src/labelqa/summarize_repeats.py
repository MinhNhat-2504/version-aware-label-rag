"""Audit exact-prompt variability in the four-question development pilot."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

from .data import read_jsonl


ROOT = Path("results")
FILES = {
    "initial_unseeded": ROOT / "version_pilot_bm25_table_responses.jsonl",
    "five_unseeded_repeats": ROOT / "version_table_v12_repeats.jsonl",
    "five_seeded_repeats": ROOT / "version_table_v12_repeats_seeded.jsonl",
}


def main() -> None:
    qids = ("version-01-v12", "version-02-v12")
    groups = defaultdict(list)
    for label, path in FILES.items():
        for row in read_jsonl(path):
            if row["qid"] in qids:
                groups[(label, row["qid"])].append(row)
    report = {"status": "development_prompt_repetition_diagnostic_not_general_reliability_estimate",
              "runs": {}}
    for label in FILES:
        result = {}
        for qid in qids:
            rows = groups[label, qid]
            if not rows:
                raise ValueError(f"Missing {qid} in {label}")
            if len({r["model_digest"] for r in rows}) != 1 or len({r["prompt_sha256"] for r in rows}) != 1:
                raise ValueError(f"Mixed model or prompt in {label}/{qid}")
            result[qid] = {
                "n": len(rows), "model_digest": rows[0]["model_digest"],
                "prompt_sha256": rows[0]["prompt_sha256"],
                "decoding_options": rows[0]["decoding_options"],
                "unique_raw_responses": len({sha256(r["raw_response"].encode()).hexdigest() for r in rows}),
                "abstain_counts": {str(flag): sum((r.get("parsed_response") or {}).get("abstain") is flag for r in rows)
                                   for flag in (True, False)},
            }
        report["runs"][label] = result
    for qid in qids:
        if len({report["runs"][label][qid]["prompt_sha256"] for label in FILES}) != 1:
            raise ValueError(f"Prompt changed across repeat groups for {qid}")
        if len({report["runs"][label][qid]["model_digest"] for label in FILES}) != 1:
            raise ValueError(f"Model changed across repeat groups for {qid}")
    report["limitation"] = "The initial unseeded mavacamten response differs from later exact-prompt repetitions. Five repetitions per setting are too few to estimate a population error rate. Both groups of five repeated outputs were internally identical, but all falsely abstained on mavacamten; an explicit seed did not fix the decision."
    output = ROOT / "version_table_repeatability.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "mavacamten_abstain": {
        label: data["version-02-v12"]["abstain_counts"] for label, data in report["runs"].items()}}))


if __name__ == "__main__":
    main()
