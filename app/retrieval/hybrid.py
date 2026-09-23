from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.ingestion.models import Evidence
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import Reranker


class HybridRetriever:
    """A lightweight hybrid retriever combining keyword overlap, BM25, and a rerank stage."""

    def __init__(self) -> None:
        self.bm25 = BM25Retriever()
        self.reranker = Reranker()

    def search(self, query: str, evidence: Iterable[Evidence], top_k: int = 5) -> list[Evidence]:
        docs = list(evidence)
        if not docs:
            return []

        query_tokens = self._normalize_query(query)
        ranked: list[tuple[float, Evidence]] = []

        for doc in docs:
            text = self._normalize_query(doc.content)
            overlap = self._keyword_overlap_score(query_tokens, text)
            exact_phrase = 1.0 if query.lower() and query.lower() in doc.content.lower() else 0.0
            score = overlap + exact_phrase
            doc.metadata["relevance_score"] = round(score, 4)
            doc.metadata["query_terms"] = query_tokens
            ranked.append((score, doc))

        bm25_results = self.bm25.search(query, docs, top_k=top_k)
        reranked = self.reranker.rerank(query, bm25_results or docs)

        hybrid = []
        seen: set[str] = set()
        for doc in reranked + [doc for _, doc in sorted(ranked, key=lambda item: item[0], reverse=True)][:top_k]:
            if doc.document_id not in seen:
                hybrid.append(doc)
                seen.add(doc.document_id)

        results = hybrid[:top_k]
        for rank, doc in enumerate(results, start=1):
            doc.metadata["retrieval_rank"] = rank

        return results if results else docs[:top_k]

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
