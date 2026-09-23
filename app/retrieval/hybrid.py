from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.config import Settings
from app.ingestion.models import Evidence
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.qdrant_store import QdrantVectorStore
from app.retrieval.reranker import Reranker


class HybridRetriever:
    """Hybrid retrieval using dense Qdrant search plus BM25 and reranking."""

    def __init__(self) -> None:
        self.bm25 = BM25Retriever()
        self.reranker = Reranker()
        settings = Settings()
        self.qdrant = QdrantVectorStore(
            collection_name=settings.qdrant_collection,
            enabled=settings.qdrant_enabled,
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_seconds,
            api_key=settings.qdrant_api_key or None,
            cloud_inference=settings.qdrant_cloud_inference,
        )

    def search(self, query: str, evidence: Iterable[Evidence], top_k: int = 5) -> list[Evidence]:
        docs = list(evidence)
        if not docs:
            return []

        qdrant_results = self.qdrant.search(query, docs, top_k=top_k) if self.qdrant.enabled else []
        qdrant_scores = {
            doc.document_id: float(doc.metadata.get("dense_score", 0.0) or 0.0)
            for doc in qdrant_results
        }

        bm25_results = self.bm25.search(query, docs, top_k=top_k)
        bm25_scores = {doc.document_id: 0.0 for doc in docs}
        for doc in bm25_results:
            bm25_scores[doc.document_id] = max(bm25_scores.get(doc.document_id, 0.0), 1.0)

        query_tokens = self._normalize_query(query)
        lexical_ranked: list[tuple[float, Evidence]] = []
        for doc in docs:
            text = self._normalize_query(doc.content)
            overlap = self._keyword_overlap_score(query_tokens, text)
            exact_phrase = 1.0 if query.lower() and query.lower() in doc.content.lower() else 0.0
            lexical_score = overlap + exact_phrase
            doc.metadata["relevance_score"] = round(lexical_score, 4)
            doc.metadata["query_terms"] = query_tokens
            lexical_ranked.append((lexical_score, doc))

        lexical_scores = {doc.document_id: score for score, doc in lexical_ranked}
        merged: list[tuple[float, Evidence]] = []
        seen: set[str] = set()
        for doc in docs:
            dense_score = qdrant_scores.get(doc.document_id, 0.0)
            bm25_score = bm25_scores.get(doc.document_id, 0.0)
            lexical_score = lexical_scores.get(doc.document_id, 0.0)
            combined = self._hybrid_score(query, doc, dense=dense_score, bm25=bm25_score, lexical=lexical_score)
            doc.metadata["dense_score"] = dense_score
            doc.metadata["bm25_score"] = bm25_score
            doc.metadata["lexical_score"] = lexical_score
            doc.metadata["hybrid_score"] = round(combined, 4)
            if doc.document_id not in seen:
                merged.append((combined, doc))
                seen.add(doc.document_id)

        merged.sort(key=lambda item: item[0], reverse=True)
        results = [doc for _, doc in merged[:top_k]]
        reranked = self.reranker.rerank(query, results if results else docs)
        final_results = reranked[:top_k]
        for rank, doc in enumerate(final_results, start=1):
            doc.metadata["retrieval_rank"] = rank
        return final_results if final_results else docs[:top_k]

    def _hybrid_score(self, query: str, doc: Evidence, *, dense: float = 0.0, bm25: float = 0.0, lexical: float = 0.0) -> float:
        lexical_score = max(float(lexical or 0.0), 0.0)
        dense_score = max(float(dense or 0.0), 0.0)
        bm25_score = max(float(bm25 or 0.0), 0.0)
        exact_phrase = 1.0 if query and query.lower() in doc.content.lower() else 0.0
        return 0.35 * dense_score + 0.35 * bm25_score + 0.30 * (lexical_score + exact_phrase)

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
