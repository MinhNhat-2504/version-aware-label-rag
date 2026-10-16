"""FDARxBench readers and a deterministic label-level evaluation split."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Label:
    set_id: str
    drug_name: str
    chunks: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    qid: str
    set_id: str
    task: str
    text: str
    gold: tuple[int, ...]
    answer: str = ""


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def portable_path(value: str | Path) -> Path:
    """Interpret persisted project-relative paths on Windows and POSIX.

    Early frozen manifests were created on Windows and therefore contain
    backslashes. Replacing separators keeps those immutable files usable on
    Linux CI without rewriting their checksummed contents.
    """
    return Path(str(value).replace("\\", "/"))


def load_labels(path: Path) -> dict[str, Label]:
    labels = {}
    for item in read_jsonl(path):
        set_id = str(item["set_id"])
        # The upstream snapshot has several rows without a usable SET ID.
        # None can be joined to benchmark questions, so keep the join explicit.
        if not set_id:
            continue
        if set_id in labels:
            raise ValueError(f"Duplicate label set_id: {set_id}")
        labels[set_id] = Label(set_id, str(item["drug_name"]), tuple(item["chunks"]))
    return labels


def load_questions(path: Path) -> list[Question]:
    questions = []
    seen = set()
    for item in read_jsonl(path):
        qid = str(item["qid"])
        if qid in seen:
            raise ValueError(f"Duplicate question qid: {qid}")
        seen.add(qid)
        questions.append(
            Question(
                qid=qid,
                set_id=str(item["set_id"]),
                task=str(item["task"]),
                text=str(item["question"]),
                # Upstream `citations` are section labels such as "2.1", not
                # passage IDs. Its retrieval corpus uses `context` indices.
                gold=tuple(sorted({
                    int(context["doc_chunk_index"])
                    for context in item.get("context", [])
                    if context.get("doc_chunk_index") is not None
                })),
                answer=str(item.get("answer", "")),
            )
        )
    return questions


def split_label(set_id: str) -> str:
    """Stable 70/15/15 split; all questions for a label stay together."""
    bucket = int.from_bytes(sha256(set_id.encode("utf-8")).digest()[:8], "big") % 100
    return "dev" if bucket < 70 else "validation" if bucket < 85 else "test"


def validate_data(labels: dict[str, Label], questions: list[Question]) -> None:
    for question in questions:
        if question.set_id not in labels:
            raise ValueError(f"Missing label for question {question.qid}")
        count = len(labels[question.set_id].chunks)
        if any(index < 0 or index >= count for index in question.gold):
            raise ValueError(f"Out-of-range citation for question {question.qid}")
        if question.task not in {"factual", "multihop", "refusal"}:
            raise ValueError(f"Unknown task {question.task} for {question.qid}")
