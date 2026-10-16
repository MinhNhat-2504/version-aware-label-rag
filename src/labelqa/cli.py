"""Run a frozen, label-level FDARxBench retrieval experiment."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from .data import load_labels, load_questions, split_label, validate_data
from .evaluate import paired_cluster_bootstrap, summarize
from .retrieval import LabelIndex


DEFAULT_SOURCE = Path("data/upstream/FDARxBench/data")


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split", choices=["dev", "validation", "test", "all"], default="dev")
    parser.add_argument("--limit", type=int, default=0, help="Smoke run only; 0 means full split")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    label_path = arguments.source / "labels" / "labels.jsonl"
    qa_path = arguments.source / "qa" / "qa.jsonl"
    labels = load_labels(label_path)
    questions = load_questions(qa_path)
    validate_data(labels, questions)
    selected = [q for q in questions if arguments.split == "all" or split_label(q.set_id) == arguments.split]
    selected = [q for q in selected if q.task != "refusal"]
    if arguments.limit:
        selected = selected[: arguments.limit]

    indexes: dict[str, LabelIndex] = {}
    outcomes = {"bm25": [], "tfidf": [], "rrf_lexical": [], "section_cover": [],
                "clause_cover": [], "ontology_cover": [], "evidence_set": [], "greedy_cover": []}
    for question in selected:
        if question.set_id not in indexes:
            indexes[question.set_id] = LabelIndex(labels[question.set_id])
        index = indexes[question.set_id]
        for method in outcomes:
            outcomes[method].append((question, index.retrieve(question.text, method=method)))

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": arguments.split,
        "limit": arguments.limit,
        "dataset": {
            "labels_sha256": digest(label_path),
            "qa_sha256": digest(qa_path),
            "label_count": len(labels),
            "question_count": len(questions),
            "task_counts": dict(Counter(q.task for q in questions)),
            "split_question_counts": dict(Counter(split_label(q.set_id) for q in questions)),
        },
        "metrics": {method: summarize(rows) for method, rows in outcomes.items()},
        "comparisons": {
            method: paired_cluster_bootstrap(outcomes["bm25"], outcomes[method])
            for method in ("tfidf", "rrf_lexical", "section_cover", "clause_cover",
                           "ontology_cover", "evidence_set", "greedy_cover")
        },
    }
    report["comparisons"]["greedy_cover_minus_ontology_cover"] = paired_cluster_bootstrap(
        outcomes["ontology_cover"], outcomes["greedy_cover"]
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
