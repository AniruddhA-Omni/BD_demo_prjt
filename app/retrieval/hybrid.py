from __future__ import annotations

from typing import Any, Iterable

from app import services
from app.config import Settings, load_settings
from app.ingestion.models import Evidence
from app.observability import traceable
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import query_terms


class HybridRetriever:
    """Dense (Qdrant) + BM25 + keyword overlap, merged with configurable weights, then reranked.

    Pipeline: candidates = top ``retrieval_candidate_k`` by weighted hybrid score → reranker → top_k.
    Every score is written into ``Evidence.metadata`` so the UI and evaluation can show why a chunk ranked where it did.
    """

    def __init__(
        self,
        vector_store: Any | None = None,
        reranker: Any | None = None,
        settings: Settings | None = None,
        weights: tuple[float, float, float] | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.vector_store = vector_store if vector_store is not None else services.get_vector_store()
        self.reranker = reranker if reranker is not None else services.get_reranker()
        self.bm25 = BM25Retriever()
        self.weights = weights or (
            self.settings.hybrid_dense_weight,
            self.settings.hybrid_bm25_weight,
            self.settings.hybrid_lexical_weight,
        )

    @property
    def qdrant(self) -> Any:
        return self.vector_store

    @traceable(name="hybrid_retrieval", run_type="retriever")
    def search(
        self,
        query: str,
        evidence: Iterable[Evidence],
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[Evidence]:
        docs = list(evidence)
        if not docs:
            return []
        if not query.strip():
            return docs[:top_k]

        # Chunks are normally indexed once at upload; this only embeds evidence that arrived some other way.
        self.vector_store.index([doc for doc in docs if not self.vector_store.is_indexed(doc.key)], session_id)
        dense_scores = self.vector_store.search(
            query,
            top_k=max(self.settings.retrieval_candidate_k, top_k),
            session_id=session_id,
            document_ids={doc.document_id for doc in docs},
        )

        raw_bm25 = self.bm25.scores(query, docs)
        token_sets = self.bm25.index(docs).token_sets  # cached with the BM25 index
        bm25_max = max(raw_bm25) if raw_bm25 and max(raw_bm25) > 0 else 1.0
        terms = query_terms(query)

        scored: list[tuple[float, int, Evidence]] = []
        seen: set[str] = set()
        for position, (doc, bm25_raw, tokens) in enumerate(zip(docs, raw_bm25, token_sets)):
            if doc.key in seen:
                continue
            seen.add(doc.key)
            dense = max(dense_scores.get(doc.key, 0.0), 0.0)
            bm25 = bm25_raw / bm25_max
            lexical = self._keyword_overlap_score(terms, tokens)
            combined = self._hybrid_score(query, doc, dense=dense, bm25=bm25, lexical=lexical)
            doc.metadata.update(
                {
                    "dense_score": round(dense, 4),
                    "bm25_score": round(bm25, 4),
                    "lexical_score": round(lexical, 4),
                    "hybrid_score": round(combined, 4),
                }
            )
            scored.append((combined, -position, doc))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        candidates = [doc for _, _, doc in scored[: max(self.settings.retrieval_candidate_k, top_k)]]
        final = self.reranker.rerank(query, candidates)[:top_k]
        for rank, doc in enumerate(final, start=1):
            doc.metadata["retrieval_rank"] = rank
        return final

    def _hybrid_score(self, query: str, doc: Evidence, *, dense: float = 0.0, bm25: float = 0.0, lexical: float = 0.0) -> float:
        dense_weight, bm25_weight, lexical_weight = self.weights
        exact_phrase = 1.0 if query.strip() and query.lower().strip() in doc.search_text.lower() else 0.0
        lexical_total = min(max(float(lexical or 0.0), 0.0) + exact_phrase, 1.0)
        return (
            dense_weight * max(float(dense or 0.0), 0.0)
            + bm25_weight * max(float(bm25 or 0.0), 0.0)
            + lexical_weight * lexical_total
        )

    def _keyword_overlap_score(self, terms: set[str], document_tokens: set[str]) -> float:
        if not terms:
            return 0.0
        return sum(1 for term in terms if term in document_tokens) / len(terms)
