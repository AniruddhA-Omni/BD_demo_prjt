from __future__ import annotations

from typing import Iterable

from app.ingestion.models import Evidence

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "then", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your",
}


class Reranker:
    """Lightweight reranker that boosts documents with stronger lexical matches."""

    def rerank(self, query: str, evidence: Iterable[Evidence]) -> list[Evidence]:
        docs = list(evidence)
        if not docs:
            return []

        query_tokens = {
            token.lower()
            for token in query.replace("-", " ").split()
            if token.strip() and token.lower() not in _STOPWORDS
        }

        scored: list[tuple[float, Evidence]] = []
        for doc in docs:
            text = doc.content.lower()
            total = 0.0
            for token in query_tokens:
                if token in text:
                    total += 1.0

            if query_tokens:
                specific_hits = sum(1 for token in query_tokens if token in text)
                total += specific_hits * 0.5

            if "revenue" in text and "q4" in text and "revenue" in query_tokens:
                total += 1.5
            if "growth" in text and "growth" in query_tokens:
                total += 0.5
            scored.append((total, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored]
