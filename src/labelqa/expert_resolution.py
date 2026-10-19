"""Let one human expert resolve an expert-vs-LLM queue without editing JSON by hand.

``sheet`` writes a spreadsheet (CSV, UTF-8 with BOM for Excel) and a side-by-side HTML
comparison for a source-review queue (``check_holdout_reviews``) or an answer-review
queue (``answer_review check``). The expert fills one ``choice`` and ``notes`` per item:

- ``expert``: keep the expert's own blind review as final;
- ``llm``: adopt the LLM annotator's review as the expert's final decision;
- ``revised``: take the row from a revised file the expert exported after comparison;
- ``reject``: exclude the item (source review only).

``sign`` turns the filled sheet into a signed ``human_expert_resolution`` queue. Every
item needs an explicit choice and notes; nothing is prefilled, and an LLM ID can never
sign. Reader files and queue identity are left untouched for the compilers to verify.

``--audit-flags`` adds a separate AI quality-control audit of the LLM annotation to the
sheet and the comparison page, labelled as AI output. The flags file names the exact
LLM annotation it audited; it is refused for any other annotation or unknown item.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import html
import io
import json
from pathlib import Path

from .data import read_jsonl
from .review_resolution import is_llm


SOURCE_CHOICES = ("expert", "llm", "revised", "reject")
ANSWER_CHOICES = ("expert", "llm", "revised")
COLUMNS = ("no", "item_id", "question", "differences", "ai_audit_flags", "expert_decision",
           "llm_decision", "choice", "notes")
AUDIT_STATUS = "ai_quality_audit_flags_not_human_review_not_gold"


def load_audit_flags(path: Path, queue: dict) -> dict:
    """Validate an AI audit of the LLM annotation and index its flags by item ID."""
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != AUDIT_STATUS or not audit.get("auditor") or not isinstance(audit.get("flags"), list):
        raise ValueError("Audit flags must declare the AI-audit status, the auditor and a flag list")
    receipts = queue.get("blind_receipts") or {}
    audited = (receipts.get("llm_annotation") or {}).get("annotation_sha256")
    if audited is None or audit.get("audited_annotation_sha256") != audited:
        raise ValueError("Audit flags were made for a different LLM annotation than this queue")
    known = {item_id(row) for row in queue["queue"]}
    by_item: dict[str, list[dict]] = {}
    for flag in audit["flags"]:
        ids = flag.get("item_ids")
        if (not isinstance(ids, list) or not ids or flag.get("priority") not in ("High", "Medium", "Low")
                or not str(flag.get("finding", "")).strip()):
            raise ValueError("Each audit flag needs item IDs, a High/Medium/Low priority and a finding")
        unknown = sorted(set(ids) - known)
        if unknown:
            raise ValueError(f"Audit flags name items absent from the queue: {unknown}")
        for key in ids:
            by_item.setdefault(key, []).append(flag)
    return {"auditor": audit["auditor"], "by_item": by_item}


def flag_summary(flags: list[dict]) -> str:
    return " | ".join(f"{flag['priority']}: {flag['finding']}" for flag in flags)


def queue_kind(queue: dict) -> str:
    rows = queue.get("queue")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Queue has no items")
    return "answer" if "review_id" in rows[0] else "source"


def item_id(row: dict) -> str:
    return row.get("review_id") or row["qid"]


def sides(queue: dict, expert_id: str) -> tuple[str, str]:
    """Return the queue keys holding the expert and the LLM review."""
    readers = queue.get("reader_ids") or []
    if is_llm(expert_id) or expert_id not in readers:
        raise ValueError("The signing expert must be the human reader of this queue, not an LLM ID")
    llm = [reader for reader in readers if is_llm(reader)]
    if len(readers) != 2 or len(llm) != 1:
        raise ValueError("Expert resolution sheets need one human expert and one LLM annotator")
    expert_side = "reader_a" if readers[0] == expert_id else "reader_b"
    return expert_side, "reader_b" if expert_side == "reader_a" else "reader_a"


def decision_summary(review: dict, kind: str) -> str:
    if kind == "answer":
        return review.get("review_status", "")
    decision = review.get("item_decision", "")
    if decision == "accept":
        answerable = {True: "answerable", False: "unanswerable"}.get(review.get("answerable_from_version"), "?")
        return f"accept / {answerable} / {review.get('version_change_judgment', '')}"
    return decision


def sheet_rows(queue: dict, expert_id: str, audit: dict | None = None) -> list[dict]:
    kind = queue_kind(queue)
    expert_side, llm_side = sides(queue, expert_id)
    flags = (audit or {}).get("by_item", {})
    rows = []
    for number, row in enumerate(queue["queue"], 1):
        rows.append({"no": number, "item_id": item_id(row),
                     "question": row[expert_side].get("question", ""),
                     "differences": "; ".join(row.get("reasons", [])),
                     "ai_audit_flags": flag_summary(flags.get(item_id(row), [])),
                     "expert_decision": decision_summary(row[expert_side], kind),
                     "llm_decision": decision_summary(row[llm_side], kind),
                     "choice": "", "notes": ""})
    return rows


def write_sheet(rows: list[dict], path: Path) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(buffer.getvalue().encode("utf-8-sig"))


def read_sheet(path: Path) -> dict[str, dict]:
    """Read a filled sheet saved by Excel as comma- or semicolon-separated UTF-8 CSV."""
    try:
        text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the sheet as 'CSV UTF-8' so Vietnamese notes are preserved") from exc
    header = text.splitlines()[0] if text else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    rows = list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
    if not rows or not {"item_id", "choice", "notes"} <= set(rows[0]):
        raise ValueError("Sheet must keep the item_id, choice and notes columns")
    by_id = {row["item_id"].strip(): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Sheet repeats an item_id")
    return by_id


def source_finals(review: dict) -> dict:
    decision = review.get("item_decision")
    if decision == "rewrite":
        raise ValueError("A 'rewrite' review cannot be final; choose another source or 'reject'")
    if decision == "reject":
        return {"final_decision": "reject"}
    return {"final_decision": "accept", "final_answer": review["independent_answer"],
            "final_answerable_from_version": review["answerable_from_version"],
            "final_claims": review["required_claims"],
            "final_acceptable_section_indices_per_claim": review["acceptable_section_indices_per_claim"],
            "final_acceptable_evidence_spans_per_claim": review["acceptable_evidence_spans_per_claim"],
            "final_version_change_judgment": review["version_change_judgment"]}


def answer_finals(review: dict) -> dict:
    from .answer_review import FIELDS
    return {f"final_{field}": review[field] for field in FIELDS}


def sign(queue: dict, sheet: dict[str, dict], expert_id: str,
         revised: dict[str, dict] | None = None) -> dict:
    """Build the signed queue; the compilers then re-verify every final field."""
    kind = queue_kind(queue)
    expert_side, llm_side = sides(queue, expert_id)
    ids = [item_id(row) for row in queue["queue"]]
    if set(sheet) != set(ids):
        raise ValueError("Sheet items differ from the queue; regenerate the sheet")
    if revised is not None and any(row.get("reviewer_id") != expert_id for row in revised.values()):
        raise ValueError("The revised file must carry the expert's own reviewer ID")
    choices = SOURCE_CHOICES if kind == "source" else ANSWER_CHOICES
    signed = json.loads(json.dumps(queue))
    for row in signed["queue"]:
        key = item_id(row)
        choice = sheet[key].get("choice", "").strip().casefold()
        notes = sheet[key].get("notes", "").strip()
        if choice not in choices:
            raise ValueError(f"Choose one of {', '.join(choices)} for {key}")
        if not notes:
            raise ValueError(f"Write a resolution note for {key}")
        if choice == "reject":
            finals = {"final_decision": "reject"}
        else:
            if choice == "revised":
                if revised is None or key not in revised:
                    raise ValueError(f"'revised' needs the item in the revised file: {key}")
                source = revised[key]
            else:
                source = row[expert_side if choice == "expert" else llm_side]
            try:
                finals = source_finals(source) if kind == "source" else answer_finals(source)
            except ValueError as exc:
                raise ValueError(f"{key}: {exc}") from exc
        row.update({"resolution_mode": "human_expert_resolution", "consensus_reviewer_ids": [],
                    "expert_resolver_id": expert_id, "adjudicator_id": "",
                    "adjudication_notes": notes, "resolution_choice": choice, **finals})
    return signed


def index_revised(path: Path, kind: str) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    key = "review_id" if kind == "answer" else "qid"
    by_id = {row[key]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Revised file repeats an item")
    return by_id


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def evidence_html(review: dict) -> str:
    parts = []
    claims = review.get("required_claims") or []
    spans = review.get("acceptable_evidence_spans_per_claim") or []
    for number, claim in enumerate(claims, 1):
        quotes = "".join(
            f"<li><span class='loc'>section {esc(span.get('section_index'))}"
            + (f", table {esc(span.get('table_index'))} row {esc(span.get('row_index'))}"
               if span.get("source_kind") == "table_row" else ", prose")
            + f"</span> &ldquo;{esc(span.get('quote'))}&rdquo;</li>"
            for span in (spans[number - 1] if number <= len(spans) else []))
        parts.append(f"<li><b>{esc(claim)}</b><ul>{quotes}</ul></li>")
    return f"<ol>{''.join(parts)}</ol>" if parts else "<p class='muted'>Không có claim.</p>"


def source_review_html(review: dict) -> str:
    return (f"<p><b>{esc(decision_summary(review, 'source'))}</b></p>"
            f"<p>{esc(review.get('independent_answer'))}</p>{evidence_html(review)}"
            f"<p class='muted'>{esc(review.get('notes'))}</p>")


def answer_review_html(review: dict) -> str:
    fields = ("review_status", "answer_complete", "all_claims_supported", "citations_at_correct_location",
              "correct_version", "refusal_appropriate", "unsafe_unsupported_claim")
    judgments = "".join(f"<li>claim {esc(j.get('claim_index'))}: {esc(j.get('support'))}, "
                        f"citation valid {esc(j.get('citation_location_valid'))} {esc(j.get('notes'))}</li>"
                        for j in review.get("claim_judgments") or [])
    table = "".join(f"<tr><td>{esc(field)}</td><td>{esc(review.get(field))}</td></tr>" for field in fields)
    return (f"<table>{table}</table><ol>{judgments}</ol>"
            f"<p class='muted'>{esc(review.get('review_notes'))}</p>")


def audit_html(flags: list[dict], auditor: str) -> str:
    if not flags:
        return ""
    items = "".join(f"<li><b>{esc(flag['priority'])}:</b> {esc(flag['finding'])}"
                    + (f"<br><span class='muted'>Gợi ý kiểm tra: {esc(flag['suggestion'])}</span>"
                       if flag.get("suggestion") else "") + "</li>" for flag in flags)
    return (f"<div class='audit'><h3>Cờ rà soát của AI ({esc(auditor)}), không phải chuyên gia</h3>"
            f"<ul>{items}</ul></div>")


def comparison_html(queue: dict, expert_id: str, audit: dict | None = None) -> str:
    kind = queue_kind(queue)
    expert_side, llm_side = sides(queue, expert_id)
    render = source_review_html if kind == "source" else answer_review_html
    flags = (audit or {}).get("by_item", {})
    auditor = (audit or {}).get("auditor", "")
    blocks = []
    for number, row in enumerate(queue["queue"], 1):
        expert, llm = row[expert_side], row[llm_side]
        context = ""
        if kind == "answer":
            context = (f"<details><summary>Câu trả lời mô hình và bằng chứng hiển thị</summary>"
                       f"<pre>{esc(json.dumps(expert.get('model_answer'), ensure_ascii=False, indent=1))}</pre>"
                       f"<pre>{esc(expert.get('visible_evidence'))}</pre></details>")
        version = f" &middot; SPL v{esc(row.get('spl_version'))}" if row.get("spl_version") else ""
        blocks.append(
            f"<section id='item-{number}'><h2>{number}. {esc(item_id(row))}{version}</h2>"
            f"<p class='q'>{esc(expert.get('question'))}</p>"
            f"<p class='diff'>Khác biệt: {esc('; '.join(row.get('reasons', [])))}</p>"
            f"{audit_html(flags.get(item_id(row), []), auditor)}{context}"
            f"<div class='cols'><div><h3>Chuyên gia (phiếu làm mù của bạn)</h3>{render(expert)}</div>"
            f"<div><h3>LLM (chú giải AI có công bố)</h3>{render(llm)}</div></div></section>")
    title = "Hòa giải của chuyên gia: thẩm định nguồn" if kind == "source" else "Hòa giải của chuyên gia: chấm câu trả lời"
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a64; --line:#d9d6cc; --card:#fff; --accent:#8a4b0f; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161614; --fg:#ecebe6; --muted:#a3a29b; --line:#3a3934; --card:#1f1f1c; --accent:#e0a060; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1200px; padding:16px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:8px; margin:16px 0; padding:12px 16px; }}
h2 {{ font-size:16px; margin:0 0 4px; }} h3 {{ font-size:14px; margin:8px 0; color:var(--accent); }}
.q {{ font-weight:600; }} .diff, .muted, .loc {{ color:var(--muted); font-size:13px; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:720px) {{ .cols {{ grid-template-columns:1fr; }} }}
pre {{ white-space:pre-wrap; font-size:12px; }} td {{ padding:2px 8px 2px 0; vertical-align:top; }}
.audit {{ margin:8px 0; padding:8px 12px; border-left:4px solid var(--accent); background:rgba(138,75,15,.08); }}
.audit h3 {{ margin:0 0 4px; }} .audit ul {{ margin:0; padding-left:20px; }}
</style></head><body>
<h1>{title}</h1>
<p>Mã chuyên gia: <b>{esc(expert_id)}</b>. Cột LLM là chú giải của AI, không phải người thẩm định.
Với mỗi mục, ghi một <code>choice</code> (expert, llm, revised hoặc reject) và một ghi chú vào bảng tính.
Gold chỉ là những gì bạn ký.</p>
{(f"<p class='audit'>Một số mục có <b>cờ rà soát của AI ({esc(auditor)})</b> về phiếu LLM. Cờ chỉ là gợi ý kiểm tra "
  "và có thể sai; hãy tự đối chiếu với nhãn.</p>") if flags else ""}
{''.join(blocks)}
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    sheet_parser = sub.add_parser("sheet", help="Write the resolution spreadsheet and comparison page")
    sheet_parser.add_argument("--queue", type=Path, required=True)
    sheet_parser.add_argument("--expert-id", required=True)
    sheet_parser.add_argument("--output-dir", type=Path, required=True)
    sheet_parser.add_argument("--audit-flags", type=Path,
                              help="Optional AI quality-control flags on the LLM annotation (JSON)")
    sign_parser = sub.add_parser("sign", help="Turn the filled spreadsheet into a signed queue")
    sign_parser.add_argument("--queue", type=Path, required=True)
    sign_parser.add_argument("--expert-id", required=True)
    sign_parser.add_argument("--sheet", type=Path, required=True)
    sign_parser.add_argument("--revised", type=Path,
                             help="Optional file the expert exported after comparison, for 'revised' items")
    sign_parser.add_argument("--audit-flags", type=Path,
                             help="The same AI audit flags file shown in the sheet, to record its hash")
    sign_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    audit = load_audit_flags(args.audit_flags, queue) if args.audit_flags else None
    if args.mode == "sheet":
        sheet, page = args.output_dir / "resolution_sheet.csv", args.output_dir / "resolution_comparison.html"
        if sheet.exists() or page.exists():
            parser.error("Resolution sheet already exists; keep the expert's filled copy")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_sheet(sheet_rows(queue, args.expert_id, audit), sheet)
        page.write_text(comparison_html(queue, args.expert_id, audit), encoding="utf-8", newline="\n")
        print(json.dumps({"sheet": str(sheet), "comparison": str(page), "items": len(queue["queue"]),
                          "flagged_items": len(audit["by_item"]) if audit else 0}))
        return
    if args.output.exists():
        parser.error("Signed queue already exists; write a new versioned path")
    revised = index_revised(args.revised, queue_kind(queue)) if args.revised else None
    signed = sign(queue, read_sheet(args.sheet), args.expert_id, revised)
    signed["expert_resolution_inputs"] = {
        "sheet_sha256": sha256(args.sheet.read_bytes()).hexdigest(),
        "revised_sha256": sha256(args.revised.read_bytes()).hexdigest() if args.revised else None,
        "audit_flags_sha256": sha256(args.audit_flags.read_bytes()).hexdigest() if args.audit_flags else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(signed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    counts: dict[str, int] = {}
    for row in signed["queue"]:
        counts[row["resolution_choice"]] = counts.get(row["resolution_choice"], 0) + 1
    print(json.dumps({"signed_queue": str(args.output), "items": len(signed["queue"]), "choices": counts}))


if __name__ == "__main__":
    main()
