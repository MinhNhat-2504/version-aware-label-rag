"""Rehearse the post-gold holdout CLIs end to end on pinned development SPL pairs.

The rehearsal builds an isolated project root, writes clearly labelled assistant
"reader" files for four development label pairs, and then runs the exact command
sequence planned for the sealed holdout: review check, gold compilation, execution
lock, prompt preparation, local model calls, retrieval summary, method-blind answer
packet, answer-review reconciliation and end-to-end scoring. It never reads the
sealed holdout. Its reader and answer judgments are synthetic plumbing inputs, so
every score it produces is a software check, not an evaluation result.

``--design llm_expert`` (the active design) uses one ``llm_`` annotator and one human
expert: blind receipt, receipt-checked queue, spreadsheet resolution signed by the
expert, the LLM answer-review converter and a no-op resume of a complete locked run.
``--design two_reader`` rehearses the earlier two-human consensus path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from .check_holdout_reviews import normalized
from .data import read_jsonl
from .spl_tables import parse_structured_passages


REPO = Path(__file__).resolve().parents[2]
DEV_XML = REPO / "results/pinned_xml_development"
DEV_SOURCES = (REPO / "data/snapshots/version_pair_paxlovid.jsonl",
               REPO / "data/snapshots/version_pairs_3.jsonl")
COPIED_FILES = ("results/learned_selector_model.json", "results/learned_selector_validation.json",
                "paper/ANALYSIS_PLAN.md")
OUTPUT = REPO / "results/holdout_pipeline_rehearsal.json"
METHODS = ("bm25", "learned_cover", "ontology_cover")
READER_DESIGNS = {"two_reader": ("rehearsal-reader-a", "rehearsal-reader-b"),
                  "llm_expert": ("llm_rehearsal_annotator", "rehearsal-expert")}
READERS = READER_DESIGNS["two_reader"]
PAXLOVID = "8a99d6d6-fd9e-45bb-b1bf-48c7f761232a"
GASTROCROM = "00623958-a8fe-47d1-a0d2-0aa4e2f7966e"
ATORVASTATIN = "01683314-5996-dbe4-e063-6394a90aa07a"
OLMESARTAN = "017bc6a3-5b3d-2381-e063-6394a90abece"
# Pairs are assigned to the two prespecified cohort file names so that the
# unchanged reporting code exercises both strata.
COHORT_FILES = {
    "data/snapshots/version_change_cohort_sources.jsonl": (PAXLOVID, GASTROCROM),
    "data/snapshots/version_holdout_sources.jsonl": (OLMESARTAN, ATORVASTATIN),
}
SUZETRIGINE_CLAIMS = [
    ("Coadministration of PAXLOVID with suzetrigine is contraindicated.",
     [(22, "table_row", "Co-administration contraindicated due to potential for serious and/or "
                        "life-threatening suzetrigine adverse reactions"),
      (11, "prose", "Non-opioid analgesic (selective blocker of Na v 1.8 sodium channels): suzetrigine")]),
]
MAVACAMTEN_CLAIMS = [
    ("Coadministration may increase mavacamten concentration and the risk of heart failure.",
     [(22, "table_row", "Co-administration with mavacamten may increase mavacamten plasma "
                        "concentration and increase the risk of heart failure.")]),
    ("Mavacamten should be discontinued for the duration of PAXLOVID treatment.",
     [(22, "table_row", "Discontinue mavacamten for the duration of PAXLOVID treatment.")]),
]
OLMESARTAN_CLAIMS = [
    ("The combination is indicated for hypertension, to lower blood pressure.",
     [(2, "prose", "is indicated for the treatment of hypertension, to lower blood pressure")]),
    ("It is not indicated for initial therapy of hypertension.",
     [(2, "prose", "is not indicated for the initial therapy of hypertension")]),
]
GASTROCROM_CLAIMS = [
    ("GASTROCROM is indicated in the management of patients with mastocytosis.",
     [(4, "prose", "GASTROCROM is indicated in the management of patients with mastocytosis.")]),
]
# question, per-version answerability and claims (None marks a rejected question).
ITEMS = [
    (PAXLOVID, "suzetrigine", "What does this PAXLOVID label say about coadministration with suzetrigine?",
     {11: [], 12: SUZETRIGINE_CLAIMS}),
    (PAXLOVID, "mavacamten", "What does this PAXLOVID label say about coadministration with mavacamten?",
     {11: [], 12: MAVACAMTEN_CLAIMS}),
    (GASTROCROM, "indication", "What is GASTROCROM indicated for?", {5: GASTROCROM_CLAIMS, 6: GASTROCROM_CLAIMS}),
    (OLMESARTAN, "indication", "What is olmesartan medoxomil and hydrochlorothiazide indicated for, "
                               "and is it indicated for initial therapy?",
     {2: OLMESARTAN_CLAIMS, 3: OLMESARTAN_CLAIMS}),
    (ATORVASTATIN, "vague", "What should be known about this atorvastatin product?", {1: None, 2: None}),
]


def locate_spans(xml_bytes: bytes, sections: dict[int, str],
                 spans: list[tuple[int, str, str]]) -> list[dict]:
    """Resolve rehearsal quotes to the parser's exact prose or table-row locator."""
    passages = parse_structured_passages(xml_bytes)
    located = []
    for section_index, kind, quote in spans:
        if normalized(quote) not in normalized(sections[section_index]):
            raise ValueError(f"Rehearsal quote absent from section {section_index}: {quote[:60]}")
        match = next((p for p in passages if p.section_index == section_index and p.kind == kind
                      and normalized(quote) in normalized(p.text)), None)
        if match is None:
            raise ValueError(f"Rehearsal quote has no {kind} passage in section {section_index}")
        located.append({"section_index": section_index, "source_kind": kind, "quote": quote,
                        "table_index": match.table_index, "row_index": match.row_index})
    return located


def build_inputs(root: Path, readers_ids: tuple[str, str] = READERS) -> dict:
    """Write the isolated sources, candidates and two synthetic reader files."""
    pairs = {row["set_id"]: row for source in DEV_SOURCES for row in read_jsonl(source)}
    for relative, set_ids in COHORT_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(pairs[set_id], ensure_ascii=False) + "\n"
                                for set_id in set_ids), encoding="utf-8")
    source_of = {set_id: relative for relative, set_ids in COHORT_FILES.items() for set_id in set_ids}
    xml_dir = root / "results/pinned_xml_holdout_after_gold"
    xml_dir.mkdir(parents=True, exist_ok=True)
    candidates, readers = [], {reader: [] for reader in readers_ids}
    for set_id, slug, question, versions in ITEMS:
        for version in pairs[set_id]["versions"]:
            number = version["spl_version"]
            xml_name = f"{set_id}_v{number}.xml"
            shutil.copyfile(DEV_XML / xml_name, xml_dir / xml_name)
            if sha256((xml_dir / xml_name).read_bytes()).hexdigest() != version["xml_sha256"]:
                raise ValueError(f"Pinned development XML changed: {xml_name}")
            candidate = {
                "qid": f"rh-{set_id[:8]}-{slug}-v{number}", "pair_id": set_id, "set_id": set_id,
                "spl_version": number, "published_date": version["published_date"],
                "xml_sha256": version["xml_sha256"], "source_url": version["source_url"],
                "source_snapshot": source_of[set_id], "question": question,
                "question_origin": "rehearsal_development_pair_not_holdout",
                "annotation_status": "candidate_pending_two_independent_readers",
            }
            candidates.append(candidate)
            claims = versions[number]
            sections = {s["section_index"]: s["text"] for s in version["sections"]}
            for position, reader in enumerate(readers_ids):
                review = {**candidate, "reviewer_id": reader, "clinical_significance": "",
                          "notes": "Synthetic rehearsal reader; not a human judgment."}
                if claims is None:
                    review.update({"item_decision": "reject", "independent_answer": "",
                                   "answerable_from_version": None, "required_claims": [],
                                   "acceptable_section_indices_per_claim": [],
                                   "acceptable_evidence_spans_per_claim": [],
                                   "version_change_judgment": ""})
                else:
                    spans = [locate_spans((xml_dir / xml_name).read_bytes(), sections, claim_spans)
                             for _, claim_spans in claims]
                    answer = (" ".join(text for text, _ in claims) if claims else
                              f"Version {number} does not address this question; no safety inference is made.")
                    # Reader B words one answer differently to exercise the disagreement queue.
                    if position == 1 and slug == "indication" and set_id == OLMESARTAN:
                        answer = "Hypertension; it is not for initial therapy."
                    review.update({
                        "item_decision": "accept", "independent_answer": answer,
                        "answerable_from_version": bool(claims),
                        "required_claims": [text for text, _ in claims],
                        "acceptable_section_indices_per_claim": [
                            sorted({span["section_index"] for span in group}) for group in spans],
                        "acceptable_evidence_spans_per_claim": spans,
                        "version_change_judgment": ("material" if set_id == PAXLOVID else "stable"),
                    })
                readers[reader].append(review)
    write_jsonl(root / "data/holdout/rehearsal_candidates.jsonl", candidates)
    for reader, rows in readers.items():
        write_jsonl(root / f"data/review/{reader}.jsonl", rows)
    for relative in COPIED_FILES:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, root / relative)
    return {"candidates": len(candidates), "pairs": len({row["pair_id"] for row in candidates})}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_llm_provenance(root: Path, reader_file: str, basis: str, basis_key: str, status: str,
                         annotator: str) -> str:
    """Synthetic provenance receipt for the rehearsal LLM file (no model wrote it)."""
    path = f"{reader_file[:-len('.jsonl')]}.provenance.json"
    receipt = {"status": status, "annotator_id": annotator, "model": "rehearsal fixture, no model call",
               "disclosures": ["Synthetic rehearsal annotation on development labels."],
               "annotation_sha256": sha256((root / reader_file).read_bytes()).hexdigest(),
               basis_key: sha256((root / basis).read_bytes()).hexdigest()}
    (root / path).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


def fill_resolution_sheet(path: Path, output: Path) -> None:
    """Rehearsal expert keeps their own review except one adopted LLM row."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig"))))
    for number, row in enumerate(rows):
        row["choice"] = "llm" if number == 0 else "expert"
        row["notes"] = "Synthetic rehearsal resolution; development data only."
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    output.write_bytes(buffer.getvalue().encode("utf-8-sig"))


def sign_gold_queue(path: Path, output: Path) -> None:
    queue = json.loads(path.read_text(encoding="utf-8"))
    for row in queue["queue"]:
        source = row["reader_a"]
        row.update({"resolution_mode": "two_reader_consensus", "consensus_reviewer_ids": list(READERS),
                    "adjudication_notes": "Synthetic rehearsal consensus; development data only."})
        if source["item_decision"] == "reject":
            row["final_decision"] = "reject"
            continue
        row.update({"final_decision": "accept", "final_answer": source["independent_answer"],
                    "final_answerable_from_version": source["answerable_from_version"],
                    "final_claims": source["required_claims"],
                    "final_acceptable_section_indices_per_claim": source["acceptable_section_indices_per_claim"],
                    "final_acceptable_evidence_spans_per_claim": source["acceptable_evidence_spans_per_claim"],
                    "final_version_change_judgment": source["version_change_judgment"]})
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def synthetic_answer_review(row: dict, reader: str, answerable: dict[str, bool]) -> dict:
    """Fill a schema-valid placeholder review; it asserts no clinical support."""
    from .answer_review import mechanical
    machine = mechanical(row)
    review = {**row, "reviewer_id": reader,
              "review_notes": "Synthetic rehearsal review; no clinical judgment was made."}
    if not machine["valid_schema"]:
        return {**review, "review_status": "invalid_output", "claim_judgments": []}
    review.update({"review_status": "assessed", "correct_version": True,
                   "unsafe_unsupported_claim": False})
    if machine["abstain"]:
        return {**review, "refusal_appropriate": not answerable[row["qid"]], "claim_judgments": []}
    judgments = [{"claim_index": index, "support": "uncertain", "citation_location_valid": True,
                  "notes": "Placeholder: rehearsal makes no support judgment."}
                 for index, _ in enumerate(row["model_answer"]["claims"])]
    return {**review, "answer_complete": False, "all_claims_supported": not judgments,
            "citations_at_correct_location": True, "claim_judgments": judgments}


def sign_answer_queue(path: Path, output: Path) -> None:
    from .answer_review import FIELDS
    queue = json.loads(path.read_text(encoding="utf-8"))
    for row in queue["queue"]:
        row.update({"resolution_mode": "two_reader_consensus", "consensus_reviewer_ids": list(READERS),
                    "adjudication_notes": "Synthetic rehearsal consensus; development data only."})
        row.update({f"final_{field}": row["reader_a"][field] for field in FIELDS})
    output.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Runner:
    def __init__(self, root: Path):
        self.root = root
        self.stages: list[dict] = []
        self.env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONIOENCODING": "utf-8"}

    def run(self, name: str, module: str, *args: str, expect_failure: bool = False) -> str:
        start = perf_counter()
        result = subprocess.run([sys.executable, "-m", f"labelqa.{module}", *args], cwd=self.root,
                                env=self.env, capture_output=True, text=True, encoding="utf-8")
        failed = result.returncode != 0
        stage = {"stage": name, "command": " ".join(["python -m", f"labelqa.{module}", *args]),
                 "expected": "refusal" if expect_failure else "success",
                 "passed": failed == expect_failure,
                 "seconds": round(perf_counter() - start, 2)}
        if failed:
            stage["error_tail"] = (result.stderr.strip().splitlines() or [""])[-1][:300]
        self.stages.append(stage)
        if not stage["passed"]:
            raise RuntimeError(f"Rehearsal stage {name} did not behave as expected:\n{result.stderr[-2000:]}")
        return result.stdout


def rehearse(root: Path, model: str, design: str = "llm_expert") -> dict:
    from .answer_review import ANSWER_BLIND_STATUS, FIELDS, LLM_ANSWER_STATUS  # noqa: F401
    from .check_holdout_reviews import LLM_PROVENANCE_STATUS
    readers = READER_DESIGNS[design]
    expert_design = design == "llm_expert"
    inputs = build_inputs(root, readers)
    run = Runner(root)
    xml = ("--xml-dir", "results/pinned_xml_holdout_after_gold")
    candidates, reader_a, reader_b = ("data/holdout/rehearsal_candidates.jsonl",
                                      f"data/review/{readers[0]}.jsonl", f"data/review/{readers[1]}.jsonl")
    queue, signed = "data/review/rehearsal_queue.json", "data/review/rehearsal_queue_signed.json"
    receipts: tuple[str, ...] = ()
    if expert_design:
        provenance = write_llm_provenance(root, reader_a, candidates, "candidates_sha256",
                                          LLM_PROVENANCE_STATUS, readers[0])
        human_receipt = f"data/review/{readers[1]}.receipt.json"
        run.run("expert_blind_receipt", "check_holdout_reviews", "--candidates", candidates,
                "--reader-a", reader_b, "--receipt", human_receipt, *xml)
        receipts = ("--human-receipt", human_receipt, "--llm-provenance", provenance)
        run.run("refuse_llm_queue_without_receipts", "check_holdout_reviews", "--candidates", candidates,
                "--reader-a", reader_a, "--reader-b", reader_b, "--output", queue, *xml,
                expect_failure=True)
    run.run("check_reader_files", "check_holdout_reviews", "--candidates", candidates,
            "--reader-a", reader_a, "--reader-b", reader_b, "--output", queue, *xml, *receipts)
    if expert_design:
        run.run("expert_resolution_sheet", "expert_resolution", "sheet", "--queue", queue,
                "--expert-id", readers[1], "--output-dir", "data/review/resolution")
        fill_resolution_sheet(root / "data/review/resolution/resolution_sheet.csv",
                              root / "data/review/resolution/filled.csv")
        run.run("expert_resolution_sign", "expert_resolution", "sign", "--queue", queue,
                "--expert-id", readers[1], "--sheet", "data/review/resolution/filled.csv", "--output", signed)
    else:
        sign_gold_queue(root / queue, root / signed)
    run.run("compile_gold", "compile_holdout_gold", "--candidates", candidates, "--reader-a", reader_a,
            "--reader-b", reader_b, "--adjudicated-queue", signed, *xml, *receipts)
    run.run("refuse_gold_overwrite", "compile_holdout_gold", "--candidates", candidates,
            "--reader-a", reader_a, "--reader-b", reader_b, "--adjudicated-queue", signed,
            *xml, *receipts, expect_failure=True)
    run.run("refuse_prompts_before_lock", "prepare_holdout_eval", "--method", "bm25",
            "--output", "results/holdout_bm25_prompts.jsonl", expect_failure=True)
    run.run("execution_lock", "lock_holdout_execution", "--model", model)
    for method in METHODS:
        run.run(f"prompts_{method}", "prepare_holdout_eval", "--method", method,
                "--output", f"results/holdout_{method}_prompts.jsonl")
    run.run("refuse_out_of_order_call", "run_local", "--input", "results/holdout_learned_cover_prompts.jsonl",
            "--output", "results/holdout_learned_cover_responses.jsonl", "--model", model,
            "--seed", "20261016", expect_failure=True)
    for method in METHODS:
        run.run(f"answers_{method}", "run_local", "--input", f"results/holdout_{method}_prompts.jsonl",
                "--output", f"results/holdout_{method}_responses.jsonl", "--model", model, "--seed", "20261016")
    run.run("refuse_repeat_call", "run_local", "--input", "results/holdout_bm25_prompts.jsonl",
            "--output", "results/holdout_bm25_responses.jsonl", "--model", model, "--seed", "20261016",
            expect_failure=True)
    before = (root / "results/holdout_bm25_responses.jsonl").read_bytes()
    resumed = run.run("resume_complete_run_makes_no_call", "run_local", "--input",
                      "results/holdout_bm25_prompts.jsonl", "--output", "results/holdout_bm25_responses.jsonl",
                      "--model", model, "--seed", "20261016", "--resume")
    if (root / "results/holdout_bm25_responses.jsonl").read_bytes() != before or \
            json.loads(resumed.strip().splitlines()[-1])["completed"] != 0:
        raise RuntimeError("Resuming a complete locked run changed its responses")
    run.run("retrieval_summary", "summarize_holdout_retrieval",
            *[arg for method in METHODS for arg in (f"--{method.replace('_', '-')}",
                                                    f"results/holdout_{method}_prompts.jsonl")],
            "--output", "results/holdout_retrieval_summary.json")
    packet, key = "review/answer_packet.jsonl", "private/answer_key.json"
    run.run("blinded_answer_packet", "prepare_blinded_answer_review", "--output", packet, "--private-map", key)
    gold = {row["qid"]: row["answerable"] for row in read_jsonl(root / "data/holdout/version_gold_adjudicated.jsonl")}
    rows = list(read_jsonl(root / packet))
    answer_files = [f"review/{reader}_answers.jsonl" for reader in readers]
    answer_receipts: tuple[str, ...] = ()
    if expert_design:
        spec = {"annotator_id": readers[0], "model": "rehearsal fixture, no model call",
                "disclosures": ["Synthetic rehearsal answer review on development labels."],
                "items": {row["review_id"]: {field: review[field] for field in FIELDS}
                          for row in rows for review in [synthetic_answer_review(row, readers[0], gold)]}}
        (root / "review/llm_answer_spec.json").write_text(json.dumps(spec), encoding="utf-8")
        run.run("llm_answer_review_converter", "build_llm_answer_review", "--spec", "review/llm_answer_spec.json",
                "--packet", packet, "--output", answer_files[0])
        write_jsonl(root / answer_files[1], [synthetic_answer_review(row, readers[1], gold) for row in rows])
        answer_receipt = f"review/{readers[1]}_answers.receipt.json"
        run.run("expert_answer_receipt", "answer_review", "receipt", "--packet", packet,
                "--reader", answer_files[1], "--output", answer_receipt)
        answer_receipts = ("--human-receipt", answer_receipt, "--llm-provenance",
                           f"{answer_files[0][:-len('.jsonl')]}.provenance.json")
    else:
        for reader, path in zip(readers, answer_files):
            write_jsonl(root / path, [synthetic_answer_review(row, reader, gold) for row in rows])
    answer_args = ("--packet", packet, "--reader-a", answer_files[0], "--reader-b", answer_files[1])
    run.run("check_answer_reviews", "answer_review", "check", *answer_args,
            "--output", "review/answer_queue.json", *answer_receipts)
    if expert_design:
        run.run("expert_answer_resolution_sheet", "expert_resolution", "sheet", "--queue",
                "review/answer_queue.json", "--expert-id", readers[1], "--output-dir", "review/resolution")
        fill_resolution_sheet(root / "review/resolution/resolution_sheet.csv", root / "review/resolution/filled.csv")
        run.run("expert_answer_resolution_sign", "expert_resolution", "sign", "--queue", "review/answer_queue.json",
                "--expert-id", readers[1], "--sheet", "review/resolution/filled.csv",
                "--output", "review/answer_queue_signed.json")
    else:
        sign_answer_queue(root / "review/answer_queue.json", root / "review/answer_queue_signed.json")
    run.run("compile_answer_reviews", "answer_review", "compile", *answer_args,
            "--adjudicated-queue", "review/answer_queue_signed.json", "--output", "review/final_answer_reviews.jsonl",
            *answer_receipts)
    run.run("end_to_end_summary", "summarize_answer_reviews", "--packet", packet, "--private-map", key,
            "--final-reviews", "review/final_answer_reviews.jsonl", "--output", "results/holdout_answer_scores.json")

    retrieval = json.loads((root / "results/holdout_retrieval_summary.json").read_text(encoding="utf-8"))
    answers = json.loads((root / "results/holdout_answer_scores.json").read_text(encoding="utf-8"))
    responses = {method: list(read_jsonl(root / f"results/holdout_{method}_responses.jsonl")) for method in METHODS}
    calls = [row for method_rows in responses.values() for row in method_rows if not row.get("error")]
    seconds = [row["total_duration_ns"] / 1e9 for row in calls if row.get("total_duration_ns")]
    prompt_tokens = [row["prompt_eval_count"] for row in calls if isinstance(row.get("prompt_eval_count"), int)]
    return {
        "design": design, "reader_ids": list(readers),
        "gold_review_design": retrieval.get("gold_review_design"),
        "answer_review_design": answers.get("answer_review_design"),
        "inputs": inputs, "stages": run.stages,
        "model_timing": {"calls": len(calls),
                         "mean_seconds_per_call": round(sum(seconds) / len(seconds), 2) if seconds else None,
                         "max_seconds_per_call": round(max(seconds), 2) if seconds else None,
                         "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
                         "context_window_tight_calls": sum(bool(row.get("context_window_tight")) for row in calls)},
        "sample_flow": retrieval["sample_flow"],
        "retrieval_success_counts": {method: sum(item[method] for item in retrieval["per_item"])
                                     for method in METHODS},
        "answerable_items": retrieval["answerable_items"],
        "table_row_gold_items": sum(any(span["source_kind"] == "table_row"
                                        for claim in row["claims"] for span in claim["acceptable_evidence_spans"])
                                    for row in read_jsonl(root / "data/holdout/version_gold_adjudicated.jsonl")),
        "model_errors": {method: sum(bool(row.get("error")) for row in rows) for method, rows in responses.items()},
        "machine_eligible_answers": {method: counts["accepted_items"] - counts["invalid_or_unusable_outputs"]
                                     for method, counts in answers["method_counts"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/holdout_rehearsal")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--design", choices=tuple(READER_DESIGNS), default="llm_expert")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.root.resolve()
    if root == REPO or REPO / "data" in root.parents or root.exists():
        parser.error("Use a new rehearsal root outside the project data directory")
    root.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        outcome = rehearse(root, args.model, args.design)
        status = "pass"
    except Exception as error:  # Record the failing stage before re-raising.
        outcome = {"error": str(error)[-2000:]}
        status = "fail"
    report = {
        "status": f"development_rehearsal_{status}_not_holdout_synthetic_reader_inputs",
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "root": root.as_posix(), "model": args.model,
        "code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        **outcome,
        "interpretation": ("Software rehearsal on development labels with synthetic reader and answer "
                           "judgments. Counts check plumbing only and are not retrieval or clinical results."),
        "holdout_results_read": 0,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "stages": len(outcome.get("stages", []))}))
    if status != "pass":
        raise SystemExit(outcome["error"])


if __name__ == "__main__":
    main()
