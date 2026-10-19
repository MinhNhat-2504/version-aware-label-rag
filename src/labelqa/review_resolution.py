"""Validate independent adjudication, signed two-reader consensus or expert resolution.

A reader ID such as ``llm_...`` or ``codex_...`` marks a disclosed AI annotator. Such an
annotator may provide one independent annotation file, but it can never sign a
consensus or act as adjudicator: when one is present, the human reader must resolve
every item (``human_expert_resolution``) or a distinct human adjudicator must do so.

Without any human reader, ``labelqa.compile_silver_standard`` applies a fixed rule to two
AI annotations; its output is an AI silver standard, never human gold.
"""

from __future__ import annotations

import re


MODES = {"independent_adjudicator", "two_reader_consensus", "human_expert_resolution"}
LLM_PREFIX = "llm_"
# AI annotator IDs: a known AI prefix, or an "ai", "llm", "codex" or "simulation" token.
AI_ID = re.compile(r"^(llm|ai|codex|gpt|claude|gemini)[_-]|(^|[_-])(ai|llm|codex|simulation)([_-]|$)",
                   re.IGNORECASE)
SILVER_MODE = "ai_silver_rule"
SILVER_STATUS = "ai_silver_standard_rule_compiled_no_human_review_not_gold"


def is_llm(reader_id: str) -> bool:
    """True for any disclosed AI annotator ID; such an ID can never sign or count as human."""
    return isinstance(reader_id, str) and bool(AI_ID.search(reader_id.strip()))


def resolution(row: dict, reader_ids: list[str] | tuple[str, str], context: str) -> tuple[str, list[str]]:
    """Return the resolution mode and signer IDs while retaining legacy queues."""
    readers = list(reader_ids)
    if len(readers) != 2 or len(set(readers)) != 2:
        raise ValueError(f"Two distinct reader IDs are required for {context}")
    llm_readers = [reader for reader in readers if is_llm(reader)]
    if len(llm_readers) > 1:
        raise ValueError(f"At most one disclosed LLM annotator is allowed for {context}")
    mode = row.get("resolution_mode")
    adjudicator = row.get("adjudicator_id")
    consensus = row.get("consensus_reviewer_ids")
    if consensus is None and mode == "two_reader_consensus":
        consensus = row.get("resolver_ids")  # Compiled records use the generic signer field.
    if not mode and isinstance(adjudicator, str) and adjudicator.strip():
        mode = "independent_adjudicator"  # Backward-compatible signed queues.
    if mode not in MODES:
        raise ValueError(f"Choose a manual resolution mode for {context}")
    if mode == "independent_adjudicator":
        if (not isinstance(adjudicator, str) or not adjudicator.strip()
                or adjudicator in readers or is_llm(adjudicator)):
            raise ValueError(f"A distinct human adjudicator ID is required for {context}")
        if consensus not in (None, []):
            raise ValueError(f"Independent adjudication cannot also claim consensus for {context}")
        return mode, [adjudicator]
    if adjudicator not in (None, ""):
        raise ValueError(f"Only independent adjudication can name a third adjudicator for {context}")
    if mode == "human_expert_resolution":
        if len(llm_readers) != 1:
            raise ValueError(f"Expert resolution is only for one human reader and one LLM annotator: {context}")
        human = next(reader for reader in readers if not is_llm(reader))
        signer = row.get("expert_resolver_id")
        if signer is None:
            signer = (row.get("resolver_ids") or [None])[0]  # Compiled records.
        if signer != human:
            raise ValueError(f"The human reader must sign the expert resolution for {context}")
        if consensus not in (None, []):
            raise ValueError(f"Expert resolution cannot also claim consensus for {context}")
        return mode, [human]
    if llm_readers:
        raise ValueError(f"An LLM annotator cannot sign a consensus for {context}")
    if (not isinstance(consensus, list) or len(consensus) != 2
            or set(consensus) != set(readers) or any(not isinstance(x, str) or not x.strip()
                                                    for x in consensus)):
        raise ValueError(f"Both original readers must sign consensus for {context}")
    return mode, readers


def compiled_status(kind: str, mode: str) -> str:
    """Manifest status naming who resolved the items; never implies automatic agreement."""
    prefix = {"gold": "", "answer": "answer_", "equivalence": "equivalence_"}[kind]
    if mode == "two_reader_consensus":
        return f"two_reader_{prefix}consensus_compiled_no_automatic_agreement_to_gold"
    if mode == "human_expert_resolution":
        return f"human_expert_{prefix}resolution_of_llm_annotation_compiled_no_automatic_agreement_to_gold"
    return f"manual_{prefix}adjudication_compiled_no_automatic_agreement_to_gold"


def review_design(manifest: dict) -> dict:
    """Who reviewed and who signed, for reports; LLM IDs are always listed separately."""
    readers = list(manifest.get("reader_ids") or [])
    llm = [reader for reader in readers if is_llm(reader)]
    if readers and len(llm) == len(readers):
        kind = "ai_ai_agreement_not_reliability"
    elif llm:
        kind = "expert_llm_agreement_not_inter_human_reliability"
    else:
        kind = "inter_reader_agreement"
    return {"resolution_mode": manifest.get("resolution_mode"),
            "resolver_ids": manifest.get("resolver_ids"),
            "reader_ids": readers, "llm_annotator_ids": llm,
            "human_review": bool(readers) and len(llm) < len(readers),
            "agreement_kind": kind}


def accepted_statuses(kind: str) -> set[str]:
    return {compiled_status(kind, mode) for mode in MODES}


def validate_compiled_resolution(row: dict, manifest: dict, reader_ids: list[str], context: str) -> None:
    """Verify a compiled row uses the same recorded resolution as its manifest."""
    mode, resolvers = resolution(row, reader_ids, context)
    if manifest.get("resolution_mode") is None and manifest.get("resolver_ids") is None:
        if mode != "independent_adjudicator":
            raise ValueError(f"Legacy manifest cannot claim consensus for {context}")
        return
    if mode != manifest.get("resolution_mode") or resolvers != manifest.get("resolver_ids"):
        raise ValueError(f"Compiled resolution differs from manifest for {context}")
