"""Prepare frozen version-holdout prompts only after manual gold exists."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from .check_holdout_reviews import span_in_passage
from .data import Label, portable_path, read_jsonl
from .holdout_reporting import sample_flow
from .learned_retrieval import FEATURE_NAMES, retrieve as learned_retrieve
from .prompts import SYSTEM_PROMPT
from .recent import parse_sections
from .retrieval import LabelIndex
from .review_resolution import accepted_statuses, validate_compiled_resolution
from .spl_tables import StructuredPassage, parse_structured_passages, structured_manifest


DEFAULT_GOLD = Path("data/holdout/version_gold_adjudicated.jsonl")
MODEL = Path("results/learned_selector_model.json")
MODEL_REPORT = Path("results/learned_selector_validation.json")
SAFE_ID = re.compile(r"^[0-9a-fA-F-]{36}$")


def load_manual_gold(path: Path) -> tuple[list[dict], dict]:
    """Fail before reading any holdout source when adjudicated gold is absent."""
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError("Manual adjudicated gold and its manifest are required; candidate questions are not gold")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in accepted_statuses("gold"):
        raise ValueError("Gold manifest is not from the manual adjudication compiler")
    if sha256(path.read_bytes()).hexdigest() != manifest.get("gold_sha256"):
        raise ValueError("Adjudicated gold checksum changed")
    rows = list(read_jsonl(path))
    if not rows or len(rows) != manifest.get("accepted") or len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("Gold item count or IDs differ from adjudication manifest")
    readers = manifest.get("reader_ids", [])
    if len(readers) != 2 or not all(isinstance(x, str) and x.strip() for x in readers) or readers[0] == readers[1]:
        raise ValueError("Two distinct reader IDs are required")
    for row in rows:
        validate_compiled_resolution(row, manifest, readers, row["qid"])
        if not row.get("adjudication_notes", "").strip():
            raise ValueError(f"Missing manual resolution notes for {row['qid']}")
        if row.get("answerable") and not row.get("claims"):
            raise ValueError(f"Answerable gold has no claims: {row['qid']}")
    sources = manifest.get("source_sha256")
    if not isinstance(sources, dict) or len(sources) != 4:
        raise ValueError("Manual gold is missing candidate and independent review receipts")
    for source, digest in sources.items():
        if not isinstance(digest, str) or sha256(portable_path(source).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Manual gold source receipt changed: {source}")
    for name, receipt in (manifest.get("blind_receipts") or {}).items():
        path = portable_path(receipt.get("receipt") or receipt.get("provenance") or "")
        expected = receipt.get("receipt_sha256") or receipt.get("provenance_sha256")
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Blind review receipt changed or is missing: {name}")
    sample_flow(rows, manifest)
    return rows, manifest


def model_weights() -> tuple[list[float], str]:
    report = json.loads(MODEL_REPORT.read_text(encoding="utf-8"))
    digest = sha256(MODEL.read_bytes()).hexdigest()
    if report.get("model_sha256") != digest:
        raise ValueError("Development selector differs from validation model")
    model = json.loads(MODEL.read_text(encoding="utf-8"))
    if model.get("split_used_for_fitting") != "dev" or model.get("features") != list(FEATURE_NAMES):
        raise ValueError("Selector has unexpected training split or features")
    weights = model.get("weights", {})
    if set(weights) != set(FEATURE_NAMES):
        raise ValueError("Selector weights have changed")
    return [weights[name] for name in FEATURE_NAMES], digest


def passage_groups(claims: list[dict], passages: list[StructuredPassage], qid: str) -> list[list[int]]:
    """Only count passages containing an adjudicated quote at its exact locator."""
    groups = []
    for claim in claims:
        accepted = set()
        for span in claim["acceptable_evidence_spans"]:
            accepted.update(passage.passage_id for passage in passages if span_in_passage(span, passage))
        if not accepted:
            raise ValueError(f"No parsed passage covers adjudicated claim span for {qid}")
        groups.append(sorted(accepted))
    return groups


def prepare(rows: list[dict], xml_dir: Path, method: str) -> list[dict]:
    if method not in ("bm25", "ontology_cover", "learned_cover"):
        raise ValueError("Unknown frozen retrieval method")
    weights, model_hash = model_weights() if method == "learned_cover" else (None, None)
    snapshots: dict[Path, dict] = {}
    indexed = {}
    output = []
    for row in rows:
        qid, set_id, version_number = row["qid"], row["set_id"], row["spl_version"]
        if not SAFE_ID.fullmatch(set_id) or type(version_number) is not int or version_number < 1:
            raise ValueError(f"Unsafe source identity: {qid}")
        source_path = portable_path(row["source_snapshot"])
        if source_path not in snapshots:
            snapshots[source_path] = {pair["set_id"]: pair for pair in read_jsonl(source_path)}
        pair = snapshots[source_path][set_id]
        source = next((v for v in pair["versions"] if v["spl_version"] == version_number), None)
        if source is None or source["xml_sha256"] != row["xml_sha256"]:
            raise ValueError(f"Pinned version mismatch: {qid}")
        key = (set_id, version_number)
        if key not in indexed:
            xml_path = xml_dir / f"{set_id}_v{version_number}.xml"
            xml_bytes = xml_path.read_bytes()
            if sha256(xml_bytes).hexdigest() != source["xml_sha256"]:
                raise ValueError(f"Pinned XML hash mismatch: {qid}")
            if parse_sections(xml_bytes) != source["sections"]:
                raise ValueError(f"Frozen source section text differs from XML: {qid}")
            passages = parse_structured_passages(xml_bytes)
            if not passages:
                raise ValueError(f"No parsed passages: {qid}")
            chunks = tuple(p.section_title + "\n" + p.text for p in passages)
            index = LabelIndex(Label(set_id, pair["title"], chunks))
            indexed[key] = (passages, index, structured_manifest(xml_bytes, passages))
        passages, index, parser_manifest = indexed[key]
        groups = passage_groups(row["claims"], passages, qid) if row["answerable"] else []
        ranking = (learned_retrieve(index, row["question"], weights, k=5, coverage=True)
                   if weights is not None else index.retrieve(row["question"], method, k=5))
        selected = [passages[position] for position in ranking]
        task = "refusal" if not row["answerable"] else ("multihop" if len(groups) > 1 else "factual")
        prompt = {
            "qid": qid, "set_id": set_id, "spl_version": version_number,
            "xml_sha256": source["xml_sha256"], "source_snapshot": str(source_path),
            "task": task, "question": row["question"], "reference_answer": row["reference_answer"],
            "schema_version": 2, "passage_scheme": "SPL prose windows plus atomic table rows",
            "parser_manifest": parser_manifest, "method": method, "selector_model_sha256": model_hash,
            "retrieved_indices": ranking,
            "passage_map": [{"passage_id": p.passage_id, "section_index": p.section_index,
                             "kind": p.kind, "table_index": p.table_index, "row_index": p.row_index}
                            for p in selected],
            "gold_indices": sorted(set().union(*map(set, groups))) if groups else [],
            "gold_claims": groups,
            "system_prompt": SYSTEM_PROMPT +
                f"\nUse only SPL version {version_number}. Absence of a named drug is not evidence of safety.",
            "user_prompt": f"Drug: {pair['title']}\nTarget SPL version: {version_number}\n"
                           f"Question: {row['question']}\n\nLabel passages:\n" +
                           "\n\n".join(
                               f"[P{p.passage_id:04d}]\nSPL section {p.section_index}: {p.section_title}"
                               f" ({'table row ' + str(p.table_index) + ':' + str(p.row_index) if p.kind == 'table_row' else 'prose'})\n{p.text}"
                               for p in selected),
        }
        output.append(prompt)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--xml-dir", type=Path, default=Path("results/pinned_xml_holdout_after_gold"))
    parser.add_argument("--execution-lock", type=Path, default=Path("paper/HOLDOUT_EXECUTION_LOCK.json"))
    parser.add_argument("--method", choices=("bm25", "ontology_cover", "learned_cover"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite frozen holdout prompts")
    rows, manifest = load_manual_gold(args.gold)
    from .lock_holdout_execution import verify_lock
    lock, lock_hash = verify_lock(args.execution_lock, rows, args.gold, manifest)
    if portable_path(lock["xml_dir"]) != args.xml_dir:
        raise ValueError("Prompt XML directory differs from the frozen execution lock")
    if args.output != portable_path(lock["prompt_paths"][args.method]):
        raise ValueError("Prompt output path differs from the frozen execution lock")
    prompts = prepare(rows, args.xml_dir, args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prompts), encoding="utf-8")
    receipt = {"status": "holdout_prompts_after_manual_gold", "method": args.method,
               "gold_sha256": manifest["gold_sha256"], "gold_manifest_sha256": sha256(args.gold.with_suffix(".manifest.json").read_bytes()).hexdigest(),
               "prompts_sha256": sha256(args.output.read_bytes()).hexdigest(), "n": len(prompts),
               "execution_lock_sha256": lock_hash,
               "model_sha256": prompts[0]["selector_model_sha256"] if prompts else None,
               "max_prompt_characters": max((len(row["system_prompt"]) + len(row["user_prompt"])
                                             for row in prompts), default=0)}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "n": len(prompts), "method": args.method}))


if __name__ == "__main__":
    main()
