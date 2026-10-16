"""Stable, overlapping word windows for long SPL sections."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    passage_id: int
    section_index: int
    title: str
    text: str
    word_start: int
    word_end: int


def split_sections(sections: list[dict], max_words: int = 160, overlap: int = 30) -> list[Passage]:
    if max_words < 1 or not 0 <= overlap < max_words:
        raise ValueError("Require max_words > overlap >= 0")
    passages = []
    for section in sections:
        words = section["text"].split()
        starts = [0] if len(words) <= max_words else range(0, len(words), max_words - overlap)
        for start in starts:
            end = min(start + max_words, len(words))
            passages.append(Passage(
                passage_id=len(passages), section_index=section["section_index"],
                title=section["title"], text=" ".join(words[start:end]),
                word_start=start, word_end=end,
            ))
            if end == len(words):
                break
    return passages
