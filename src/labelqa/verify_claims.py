"""Conservative entity-local contradiction flags for the version development pilot.

This rule is only a review triage signal, never a clinical support judgment.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from .data import read_jsonl
from .spl_passages import split_sections
from .spl_tables import parse_structured_passages


CONTRAINDICATED = re.compile(r"\bcontraindicat\w*\b", re.I)


def contraindication_flag(claim: str, entity: str, cited_texts: list[str]) -> str:
    """Flag an explicit contraindication claim lacking a local source statement."""
    if not CONTRAINDICATED.search(claim):
        return "rule_not_applicable"
    if not cited_texts:
        return "flag_no_citation"
    entity_pattern = re.compile(rf"\b{re.escape(entity)}\b", re.I)
    for passage in cited_texts:
        for match in entity_pattern.finditer(passage):
            prefix = passage[max(0, match.start() - 300):match.start()]
            left = match.start() - len(prefix) + prefix.rfind(".") + 1 if "." in prefix else max(0, match.start() - 100)
            following = passage[match.end():match.end() + 400]
            periods = [m.start() for m in re.finditer(r"\.", following)]
            right = match.end() + (periods[1] + 1 if len(periods) > 1 else len(following))
            local = passage[left:right]
            if CONTRAINDICATED.search(local):
                return "not_flagged_local_keyword_only"
    return "flag_no_entity_local_contraindication"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/snapshots/version_pair_paxlovid.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("data/curated/version_pilot_paxlovid.jsonl"))
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--xml-dir", type=Path, default=Path("results/pinned_xml_development"))
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pair = next(read_jsonl(args.source))
    questions = {row["qid"]: row for row in read_jsonl(args.questions)}
    prompts = {row["qid"]: row for row in read_jsonl(args.prompts)}
    versions = {row["spl_version"]: row for row in pair["versions"]}
    passage_maps = {}
    results = []
    for response in read_jsonl(args.responses):
        qid = response["qid"]
        question = questions[qid]
        prompt = prompts[qid]
        version = question["spl_version"]
        if question["xml_sha256"] != versions[version]["xml_sha256"]:
            raise ValueError(f"Source mismatch for {qid}")
        expected_prompt_hash = sha256((prompt["system_prompt"] + prompt["user_prompt"]).encode()).hexdigest()
        if response.get("prompt_sha256") != expected_prompt_hash:
            raise ValueError(f"Prompt/response hash mismatch for {qid}")
        scheme = prompt.get("passage_scheme", "160-word section windows")
        key = (version, scheme)
        if key not in passage_maps:
            if scheme == "SPL prose windows plus atomic table rows":
                path = args.xml_dir / f"{pair['set_id']}_v{version}.xml"
                xml_bytes = path.read_bytes()
                if sha256(xml_bytes).hexdigest() != versions[version]["xml_sha256"]:
                    raise ValueError(f"Pinned XML mismatch: {path}")
                passages = parse_structured_passages(xml_bytes)
            elif scheme == "160-word section windows":
                passages = split_sections(versions[version]["sections"], max_words=160)
            else:
                raise ValueError(f"Unknown passage scheme: {scheme}")
            passage_maps[key] = {p.passage_id: p.text for p in passages}
        parsed = response.get("parsed_response") or {}
        for claim in parsed.get("claims", []):
            ids = []
            for citation in claim.get("citations", []):
                if isinstance(citation, str) and re.fullmatch(r"P\d{4}", citation):
                    ids.append(int(citation[1:]))
            cited = [passage_maps[key][pid] for pid in ids if pid in passage_maps[key]]
            status = contraindication_flag(claim.get("text", ""),
                                             question["absence_check"]["term"], cited)
            results.append({"qid": qid, "spl_version": version, "claim": claim.get("text", ""),
                            "passage_scheme": scheme, "citations": claim.get("citations", []), "status": status})
    counts = Counter(item["status"] for item in results)
    report = {"scope": "PAXLOVID development pilot; prompt-matched passage scheme",
              "rule": "Explicit contraindication keyword in a claim must occur in the named drug's local sentence or immediately following sentence in a cited passage. A nonflag is not proof of support.",
              "source": str(args.responses), "claims": len(results),
              "status_counts": dict(counts), "items": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"claims": len(results), "status_counts": dict(counts),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
