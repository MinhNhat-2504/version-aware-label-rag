"""Dependency-free lexical baselines and a section-aware evidence set selector."""

from __future__ import annotations

from collections import Counter
import math
import re

from .data import Label


TOKEN = re.compile(r"\b\w+\b", re.UNICODE)
CLAUSE_BOUNDARY = re.compile(
    r",\s+and\s+|\band\s+(?=what|how|why|which|whether|when)\b|\bdespite\b",
    re.IGNORECASE,
)
SECTION_INTENTS = (
    (re.compile(r"\b(strength|formulation|dosage form)s?\b", re.I), ("DOSAGE FORMS", "STRENGTHS")),
    (re.compile(r"\b(dosage|dose|dosing|regimen|administ\w*)\b", re.I), ("DOSAGE AND ADMINISTRATION",)),
    (re.compile(r"\b(interact\w*|coadminist\w*|concomitant)\b", re.I), ("DRUG INTERACTIONS",)),
    (re.compile(r"\b(adverse|side effect|suicid\w*|serious event)\b", re.I), ("ADVERSE REACTIONS", "WARNINGS")),
    (re.compile(r"\b(food|absorption|bioavailab\w*|pharmacokinetic\w*|half.life|metaboli\w*)\b", re.I), ("PHARMACOKINETICS", "CLINICAL PHARMACOLOGY")),
    (re.compile(r"\b(nurs\w*|lactat\w*|breastfeed\w*)\b", re.I), ("LACTATION", "NURSING")),
    (re.compile(r"\b(pregnan\w*|fetal|foetal)\b", re.I), ("PREGNANCY",)),
    (re.compile(r"\b(indicat\w*|treat\w*|therapeutic use)\b", re.I), ("INDICATIONS AND USAGE",)),
    (re.compile(r"\b(efficacy|clinical trial|study data)\b", re.I), ("CLINICAL STUDIES",)),
    (re.compile(r"\b(contraindicat\w*|hypersensitiv\w*)\b", re.I), ("CONTRAINDICATIONS",)),
    (re.compile(r"\b(overdos\w*|overdose)\b", re.I), ("OVERDOSAGE",)),
    (re.compile(r"\b(mechanism of action)\b", re.I), ("MECHANISM OF ACTION", "CLINICAL PHARMACOLOGY")),
)

# Question facets map to SPL headings. These rules use label text only; no gold
# passage indices or downstream answer labels enter selection.
EVIDENCE_FACETS = (
    (re.compile(r"\b(dose|dosage|dosing|regimen|schedule|infusion|administ\w*)\b", re.I),
     re.compile(r"\b(DOSAGE|ADMINISTRATION)\b", re.I),
     re.compile(r"\b(OVERDOSAGE|DOSAGE FORMS)\b", re.I)),
    (re.compile(r"\b(ecg|electrocardio\w*|qt|cardiovascular monitor\w*)\b", re.I),
     re.compile(r"ELECTROCARDIO|CARDIAC|QT", re.I), None),
    (re.compile(r"\b(indicat\w*|approved for|used to treat)\b", re.I),
     re.compile(r"INDICATIONS|USAGE", re.I), None),
    (re.compile(r"\b(contraindicat\w*|should not (?:be )?used)\b", re.I),
     re.compile(r"CONTRAINDICATIONS", re.I), None),
    (re.compile(r"\b(renal|kidney|creatinine|gfr)\b", re.I),
     re.compile(r"RENAL|LABORATORY TESTING", re.I), None),
    (re.compile(r"\b(hepatic|liver|child.pugh)\b", re.I),
     re.compile(r"HEPATIC|LIVER", re.I), None),
    (re.compile(r"\b(rash|dermatologic|stevens.johnson)\b", re.I),
     re.compile(r"RASH|DERMATOLOGIC", re.I), None),
    (re.compile(r"\b(hypoglyc\w*)\b", re.I),
     re.compile(r"HYPOGLYC", re.I), None),
    (re.compile(r"\b(interact\w*|coadminist\w*|concomitant)\b", re.I),
     re.compile(r"INTERACTIONS", re.I), None),
)
# Frozen after a dev-only 45-point grid (results/coverage_grid_dev.json).
GREEDY_COVER_WEIGHTS = (0.25, 0.3, 0.1)
NEED_BOUNDARY = re.compile(r"\s*,?\s+and\s+", re.I)
STOP = frozenset(
    "a an and are as at be by can do does for from how in is it of on or the to was "
    "were what when where which who why with drug label tablet tablets patients patient "
    "treatment use used using mg".split()
)


def tokenize(text: str) -> list[str]:
    return [word for word in TOKEN.findall(text.lower()) if len(word) > 1 and word not in STOP]


class LabelIndex:
    def __init__(self, label: Label):
        self.label = label
        self.terms = [tokenize(chunk) for chunk in label.chunks]
        self.titles = [tokenize(chunk.split("\n", 1)[0]) for chunk in label.chunks]
        self.title_lines = [chunk.split("\n", 1)[0].upper() for chunk in label.chunks]
        self.counts = [Counter(terms) for terms in self.terms]
        self.lengths = [len(terms) for terms in self.terms]
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        document_frequency = Counter()
        for terms in self.terms:
            document_frequency.update(set(terms))
        n = len(self.terms)
        self.idf = {
            term: math.log1p((n - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.tfidf_idf = {
            term: 1.0 + math.log((n + 1) / (frequency + 1))
            for term, frequency in document_frequency.items()
        }
        self.tfidf_norms = [math.sqrt(sum(
            ((1.0 + math.log(frequency)) * self.tfidf_idf[term]) ** 2
            for term, frequency in count.items())) or 1.0 for count in self.counts]

    def bm25_scores(self, question: str) -> list[float]:
        query = set(tokenize(question))
        scores = []
        for count, length in zip(self.counts, self.lengths):
            score = 0.0
            for term in query:
                frequency = count.get(term, 0)
                if frequency:
                    normalizer = frequency + 1.5 * (
                        1 - 0.75 + 0.75 * length / max(self.average_length, 1)
                    )
                    score += self.idf[term] * frequency * 2.5 / normalizer
            scores.append(score)
        return scores

    def tfidf_scores(self, question: str) -> list[float]:
        """Smoothed TF-IDF cosine baseline within a pinned label."""
        query = Counter(tokenize(question))
        weights = {term: (1.0 + math.log(frequency)) * self.tfidf_idf.get(term, 0.0)
                   for term, frequency in query.items()}
        norm = math.sqrt(sum(weight * weight for weight in weights.values())) or 1.0
        return [sum(weights.get(term, 0.0) * (1.0 + math.log(frequency)) * self.tfidf_idf[term]
                    for term, frequency in count.items() if term in weights)
                / (norm * document_norm)
                for count, document_norm in zip(self.counts, self.tfidf_norms)]

    def greedy_cover(
        self, question: str, ranking: list[int], k: int,
        novelty_weight: float, title_weight: float, diversity_weight: float,
        scores: list[float] | None = None,
    ) -> list[int]:
        """Rerank a lexical pool by marginal query coverage and section diversity."""
        scores = scores if scores is not None else self.bm25_scores(question)
        candidates = ranking[: max(20, 4 * k)]
        query = set(tokenize(question))
        weights = {term: self.idf.get(term, 0.0) for term in query}
        total = sum(weights.values()) or 1.0
        maximum = max(scores, default=0.0) or 1.0
        selected: list[int] = []
        covered: set[str] = set()
        selected_titles: set[str] = set()
        while candidates and len(selected) < k:
            def value(index: int) -> float:
                present = query & set(self.terms[index])
                title = self.title_lines[index]
                new_coverage = sum(weights[term] for term in present - covered) / total
                title_coverage = sum(weights[term] for term in query & set(self.titles[index])) / total
                different_title = float(title not in selected_titles)
                return (scores[index] / maximum + novelty_weight * new_coverage
                        + title_weight * title_coverage + diversity_weight * different_title)

            best = max(candidates, key=lambda index: (value(index), -index))
            selected.append(best)
            covered.update(query & set(self.terms[best]))
            selected_titles.add(self.title_lines[best])
            candidates.remove(best)
        return selected

    def evidence_set(self, question: str, ranking: list[int], k: int) -> list[int]:
        """Reserve passages for distinct question facets, then fill by BM25."""
        parts = NEED_BOUNDARY.split(question, maxsplit=2)
        needs = parts if len(parts) > 1 and all(len(tokenize(part)) >= 2 for part in parts) else [question]
        selected: list[int] = []
        for need in needs:
            scores = self.bm25_scores(need)
            maximum = max(scores, default=0.0) or 1.0
            facets = [(heading, excluded) for trigger, heading, excluded in EVIDENCE_FACETS
                      if trigger.search(need)]
            for heading, excluded in facets:
                candidates = [index for index, title in enumerate(self.title_lines)
                              if heading.search(title) and not (excluded and excluded.search(title))
                              and index not in selected]
                if not candidates:
                    continue
                def value(index: int) -> float:
                    title = self.title_lines[index]
                    preferred_dose = (heading is EVIDENCE_FACETS[0][1] and
                                      bool(re.search(r"RECOMMENDED DOSAGE|DOSAGE INFORMATION", title)))
                    title_terms = set(self.titles[index])
                    need_terms = set(tokenize(need))
                    title_overlap = len(title_terms & need_terms) / max(len(need_terms), 1)
                    return scores[index] / maximum + 0.25 * title_overlap + 0.40 * preferred_dose
                best = max(candidates, key=lambda index: (value(index), -index))
                selected.append(best)
                if len(selected) == k:
                    return selected
            if not facets:
                next_best = next((index for index in sorted(range(len(scores)),
                                 key=lambda index: (-scores[index], index)) if index not in selected), None)
                if next_best is not None and scores[next_best] > 0:
                    selected.append(next_best)
                    if len(selected) == k:
                        return selected
        selected.extend(index for index in ranking if index not in selected)
        return selected[:k]

    def retrieve(self, question: str, method: str = "bm25", k: int = 5) -> list[int]:
        if method not in {"bm25", "tfidf", "rrf_lexical", "section_cover", "clause_cover", "ontology_cover", "evidence_set", "greedy_cover"}:
            raise ValueError(f"Unknown retrieval method: {method}")
        if method in {"tfidf", "rrf_lexical"}:
            tfidf = self.tfidf_scores(question)
            tfidf_ranking = sorted(range(len(tfidf)), key=lambda index: (-tfidf[index], index))
            if method == "tfidf":
                return tfidf_ranking[:k]
            bm25 = self.bm25_scores(question)
            bm25_ranking = sorted(range(len(bm25)), key=lambda index: (-bm25[index], index))
            bm25_rank = {index: rank for rank, index in enumerate(bm25_ranking, 1)}
            tfidf_rank = {index: rank for rank, index in enumerate(tfidf_ranking, 1)}
            return sorted(range(len(tfidf)), key=lambda index: (
                -(1 / (60 + bm25_rank[index]) + 1 / (60 + tfidf_rank[index])), index))[:k]
        scores = self.bm25_scores(question)
        ranking = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        if method == "bm25":
            return ranking[:k]
        if method == "greedy_cover":
            return self.greedy_cover(question, ranking, k, *GREEDY_COVER_WEIGHTS, scores=scores)
        if method == "evidence_set":
            return self.evidence_set(question, ranking, k)
        if method == "ontology_cover":
            selected = ranking[:1]
            for trigger, headings in SECTION_INTENTS:
                if not trigger.search(question):
                    continue
                matches = [
                    index for index in ranking
                    if any(heading in self.title_lines[index] for heading in headings)
                ]
                if matches:
                    best = next((index for index in matches if index not in selected), None)
                    if best is not None:
                        selected.append(best)
                if len(selected) == k:
                    break
            selected.extend(index for index in ranking if index not in selected)
            return selected[:k]
        if method == "clause_cover":
            clauses = [part for part in CLAUSE_BOUNDARY.split(question) if len(tokenize(part)) >= 2]
            if len(clauses) < 2:
                return ranking[:k]
            selected = []
            # Reserve one slot for the strongest passage for each question clause.
            for clause in clauses[: min(k, 3)]:
                clause_scores = self.bm25_scores(clause)
                for index in sorted(range(len(clause_scores)), key=lambda i: (-clause_scores[i], i)):
                    if index not in selected:
                        selected.append(index)
                        break
            selected.extend(index for index in ranking if index not in selected)
            return selected[:k]

        # The candidate pool is bounded by BM25 to avoid selecting unrelated sections.
        candidates = ranking[: max(20, 4 * k)]
        query = set(tokenize(question))
        weights = {term: self.idf.get(term, 0.0) for term in query}
        total_weight = sum(weights.values()) or 1.0
        maximum = max(scores, default=0.0) or 1.0
        selected: list[int] = []
        covered: set[str] = set()
        while candidates and len(selected) < k:
            def value(index: int) -> float:
                terms = set(self.terms[index])
                title = set(self.titles[index])
                new_coverage = sum(weights.get(term, 0.0) for term in (query & terms) - covered)
                title_coverage = sum(weights.get(term, 0.0) for term in query & title)
                return (
                    0.70 * scores[index] / maximum
                    + 0.20 * new_coverage / total_weight
                    + 0.10 * title_coverage / total_weight
                )

            best = max(candidates, key=lambda index: (value(index), -index))
            selected.append(best)
            covered.update(query & set(self.terms[best]))
            candidates.remove(best)
        return selected
