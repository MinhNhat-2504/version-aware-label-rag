"""Validate either independent adjudication or signed two-reader consensus."""

from __future__ import annotations


MODES = {"independent_adjudicator", "two_reader_consensus"}


def resolution(row: dict, reader_ids: list[str] | tuple[str, str], context: str) -> tuple[str, list[str]]:
    """Return the resolution mode and signer IDs while retaining legacy queues."""
    readers = list(reader_ids)
    if len(readers) != 2 or len(set(readers)) != 2:
        raise ValueError(f"Two distinct reader IDs are required for {context}")
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
                or adjudicator in readers):
            raise ValueError(f"A distinct adjudicator ID is required for {context}")
        if consensus not in (None, []):
            raise ValueError(f"Independent adjudication cannot also claim consensus for {context}")
        return mode, [adjudicator]
    if adjudicator not in (None, ""):
        raise ValueError(f"Two-reader consensus cannot name a third adjudicator for {context}")
    if (not isinstance(consensus, list) or len(consensus) != 2
            or set(consensus) != set(readers) or any(not isinstance(x, str) or not x.strip()
                                                    for x in consensus)):
        raise ValueError(f"Both original readers must sign consensus for {context}")
    return mode, readers


def validate_compiled_resolution(row: dict, manifest: dict, reader_ids: list[str], context: str) -> None:
    """Verify a compiled row uses the same recorded resolution as its manifest."""
    mode, resolvers = resolution(row, reader_ids, context)
    if manifest.get("resolution_mode") is None and manifest.get("resolver_ids") is None:
        if mode != "independent_adjudicator":
            raise ValueError(f"Legacy manifest cannot claim consensus for {context}")
        return
    if mode != manifest.get("resolution_mode") or resolvers != manifest.get("resolver_ids"):
        raise ValueError(f"Compiled resolution differs from manifest for {context}")
