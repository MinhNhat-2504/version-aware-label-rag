"""Retrieval metrics; no LLM judging or clinical correctness claims."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random

from .data import Question


@dataclass
class Totals:
    count: int = 0
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    all_gold_at_5: float = 0.0
    all_gold_eligible: int = 0
    reciprocal_rank: float = 0.0

    def add(self, gold: tuple[int, ...], ranking: list[int]) -> None:
        expected = set(gold)
        if not expected:
            return
        self.count += 1
        self.recall_at_1 += len(expected & set(ranking[:1])) / len(expected)
        self.recall_at_5 += len(expected & set(ranking[:5])) / len(expected)
        if len(expected) <= 5:
            self.all_gold_eligible += 1
            self.all_gold_at_5 += float(expected <= set(ranking[:5]))
        self.reciprocal_rank += next(
            (1 / rank for rank, index in enumerate(ranking, 1) if index in expected), 0.0
        )

    def report(self) -> dict:
        if not self.count:
            return {"n": 0}
        return {
            "n": self.count,
            "recall@1": round(self.recall_at_1 / self.count, 4),
            "recall@5": round(self.recall_at_5 / self.count, 4),
            "all_gold@5": round(self.all_gold_at_5 / self.all_gold_eligible, 4)
            if self.all_gold_eligible else None,
            "all_gold_eligible": self.all_gold_eligible,
            "mrr@5": round(self.reciprocal_rank / self.count, 4),
        }


def summarize(rows: list[tuple[Question, list[int]]]) -> dict:
    groups = defaultdict(Totals)
    for question, ranking in rows:
        if question.task == "refusal":
            continue  # Empty gold is not a retrieval-success example.
        groups["all"].add(question.gold, ranking)
        groups[question.task].add(question.gold, ranking)
    return {name: totals.report() for name, totals in sorted(groups.items())}


def paired_cluster_bootstrap(
    baseline: list[tuple[Question, list[int]]],
    candidate: list[tuple[Question, list[int]]],
    task: str = "multihop",
    draws: int = 2000,
) -> dict:
    """95% CI for all-gold@5 difference, resampling labels, not questions."""
    if len(baseline) != len(candidate):
        raise ValueError("Paired runs have different lengths")
    by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (q_base, rank_base), (q_new, rank_new) in zip(baseline, candidate):
        if q_base.qid != q_new.qid:
            raise ValueError("Paired runs are in different question order")
        if q_base.task != task or not q_base.gold or len(q_base.gold) > 5:
            continue
        gold = set(q_base.gold)
        by_label[q_base.set_id].append(
            (float(gold <= set(rank_base[:5])), float(gold <= set(rank_new[:5])))
        )
    labels = sorted(by_label)
    if not labels:
        return {"task": task, "n_labels": 0, "n_questions": 0}
    observations = [pair for label in labels for pair in by_label[label]]
    delta = sum(new - base for base, new in observations) / len(observations)
    generator = random.Random(20261015)
    samples = []
    for _ in range(draws):
        chosen = generator.choices(labels, k=len(labels))
        numerator = sum(new - base for label in chosen for base, new in by_label[label])
        denominator = sum(len(by_label[label]) for label in chosen)
        samples.append(numerator / denominator)
    samples.sort()
    return {
        "task": task,
        "metric": "all_gold@5",
        "n_labels": len(labels),
        "n_questions": len(observations),
        "candidate_minus_baseline": round(delta, 4),
        "ci95_label_bootstrap": [
            round(samples[int(0.025 * draws)], 4),
            round(samples[min(int(0.975 * draws), draws - 1)], 4),
        ],
        "draws": draws,
        "seed": 20261015,
    }
