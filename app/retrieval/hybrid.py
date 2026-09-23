from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.ingestion.models import Evidence


class HybridRetriever:
    """A lightweight hybrid retriever that combines keyword overlap and text scoring."""

    def search(self, query: str, evidence: Iterable[Evidence], top_k: int = 5) -> list[Evidence]:
        query_tokens = self._normalize_query(query)
        ranked: list[tuple[float, Evidence]] = []

        for doc in evidence:
            text = self._normalize_query(doc.content)
            overlap = self._keyword_overlap_score(query_tokens, text)
            exact_phrase = 1.0 if query.lower() and query.lower() in doc.content.lower() else 0.0
            score = overlap + exact_phrase
            doc.metadata["relevance_score"] = round(score, 4)
            doc.metadata["query_terms"] = query_tokens
            ranked.append((score, doc))

        ranked.sort(key=lambda item: item[0], reverse=True)
        results = [doc for _, doc in ranked[:top_k]]
        for rank, doc in enumerate(results, start=1):
            doc.metadata["retrieval_rank"] = rank

        return results if results else list(evidence)[:top_k]

    def _normalize_query(self, text: str) -> list[str]:
        if not text:
            return []
        return [token for token in text.lower().replace("-", " ").split() if token.isalnum()]

    def _keyword_overlap_score(self, query_tokens: list[str], document_tokens: list[str]) -> float:
        if not query_tokens:
            return 1.0
        query_counter = Counter(query_tokens)
        document_counter = Counter(document_tokens)
        matches = sum(min(query_counter[token], document_counter[token]) for token in query_counter)
        return matches / max(len(query_counter), 1)
