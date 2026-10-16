"""Small dependency-free passage reranker trained on development labels only."""

from __future__ import annotations

import math
import random

from .retrieval import CLAUSE_BOUNDARY, SECTION_INTENTS, LabelIndex, tokenize


FEATURE_NAMES = (
    "bm25_relative", "bm25_reciprocal_rank", "tfidf_relative",
    "tfidf_reciprocal_rank", "query_term_coverage", "title_term_coverage",
    "section_intent_match", "best_clause_relative", "short_passage_preference",
)
POOL_SIZE = 20
EPOCHS = 8
LEARNING_RATE = 0.03
SEED = 20261015


def feature_pool(index: LabelIndex, question: str) -> tuple[dict[int, tuple[float, ...]], list[int]]:
    bm25 = index.bm25_scores(question)
    tfidf = index.tfidf_scores(question)
    bm25_rank = sorted(range(len(bm25)), key=lambda i: (-bm25[i], i))
    tfidf_rank = sorted(range(len(tfidf)), key=lambda i: (-tfidf[i], i))
    ontology = index.retrieve(question, "ontology_cover", 5)
    pool = list(dict.fromkeys(bm25_rank[:POOL_SIZE] + tfidf_rank[:POOL_SIZE] + ontology))
    bm25_position = {candidate: rank for rank, candidate in enumerate(bm25_rank, 1)}
    tfidf_position = {candidate: rank for rank, candidate in enumerate(tfidf_rank, 1)}
    max_bm25 = max(bm25, default=0.0) or 1.0
    max_tfidf = max(tfidf, default=0.0) or 1.0
    query_terms = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query_terms) or 1.0
    intents = [headings for trigger, headings in SECTION_INTENTS if trigger.search(question)]
    clauses = [part for part in CLAUSE_BOUNDARY.split(question) if len(tokenize(part)) >= 2]
    clause_scores = [index.bm25_scores(clause) for clause in clauses[:3]] if len(clauses) > 1 else []
    clause_maxima = [max(scores, default=0.0) or 1.0 for scores in clause_scores]
    longest = math.log1p(max(index.lengths, default=0)) or 1.0
    features = {}
    for candidate in pool:
        terms = set(index.terms[candidate])
        title_terms = set(index.titles[candidate])
        title = index.title_lines[candidate]
        features[candidate] = (
            bm25[candidate] / max_bm25,
            1 / bm25_position[candidate],
            tfidf[candidate] / max_tfidf,
            1 / tfidf_position[candidate],
            sum(index.idf.get(term, 0.0) for term in query_terms & terms) / denominator,
            sum(index.idf.get(term, 0.0) for term in query_terms & title_terms) / denominator,
            float(any(any(heading in title for heading in headings) for headings in intents)),
            max((scores[candidate] / maximum for scores, maximum in zip(clause_scores, clause_maxima)),
                default=0.0),
            1.0 - math.log1p(index.lengths[candidate]) / longest,
        )
    return features, pool


def score(weights: list[float], features: tuple[float, ...]) -> float:
    return sum(weight * value for weight, value in zip(weights, features))


def train(records: list[tuple[dict[int, tuple[float, ...]], list[int], set[int]]],
          disabled_features: frozenset[str] = frozenset()) -> tuple[list[float], dict]:
    """Pairwise logistic ranking; gold passage indices appear only in this call."""
    weights = [0.0] * len(FEATURE_NAMES)
    weights[0] = 1.0  # Start from the BM25 ordering.
    disabled_indices = {i for i, name in enumerate(FEATURE_NAMES) if name in disabled_features}
    if disabled_features - set(FEATURE_NAMES):
        raise ValueError("Unknown disabled feature")
    if 0 in disabled_indices:
        raise ValueError("BM25 anchor cannot be disabled in this ablation")
    generator = random.Random(SEED)
    eligible = [record for record in records if record[2] & set(record[1])]
    for epoch in range(EPOCHS):
        generator.shuffle(eligible)
        rate = LEARNING_RATE / (1 + 0.2 * epoch)
        for features, pool, gold in eligible:
            positives = [candidate for candidate in pool if candidate in gold]
            negatives = [candidate for candidate in pool if candidate not in gold]
            if not negatives:
                continue
            hard = negatives[:8]
            extra = generator.sample(negatives[8:], min(3, len(negatives[8:])))
            for positive in positives:
                for negative in hard + extra:
                    difference = [a - b for a, b in zip(features[positive], features[negative])]
                    margin = max(-40.0, min(40.0, sum(w * d for w, d in zip(weights, difference))))
                    gradient = 1.0 / (1.0 + math.exp(margin))
                    for j, value in enumerate(difference):
                        if j not in disabled_indices:
                            weights[j] += rate * (gradient * value - 0.0001 * weights[j])
    return weights, {"training_questions": len(records), "questions_with_a_gold_in_pool": len(eligible),
                     "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "seed": SEED,
                     "features": list(FEATURE_NAMES),
                     "disabled_features": sorted(disabled_features),
                     "pool_size_per_lexical_method": POOL_SIZE}


def retrieve(index: LabelIndex, question: str, weights: list[float], k: int = 5,
             coverage: bool = False) -> list[int]:
    features, pool = feature_pool(index, question)
    if not coverage:
        return sorted(pool, key=lambda candidate: (-score(weights, features[candidate]), candidate))[:k]
    query_terms = set(tokenize(question))
    denominator = sum(index.idf.get(term, 0.0) for term in query_terms) or 1.0
    selected = []
    covered = set()
    titles = set()
    remaining = pool.copy()
    while remaining and len(selected) < k:
        def value(candidate: int) -> float:
            new_terms = (query_terms & set(index.terms[candidate])) - covered
            novelty = sum(index.idf.get(term, 0.0) for term in new_terms) / denominator
            diversity = float(index.title_lines[candidate] not in titles)
            return score(weights, features[candidate]) + 0.12 * novelty + 0.04 * diversity
        best = max(remaining, key=lambda candidate: (value(candidate), -candidate))
        selected.append(best)
        covered.update(query_terms & set(index.terms[best]))
        titles.add(index.title_lines[best])
        remaining.remove(best)
    return selected
