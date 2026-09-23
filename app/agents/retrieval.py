from __future__ import annotations

from typing import Iterable

from app.ingestion.models import Evidence
from app.retrieval.hybrid import HybridRetriever


class RetrievalAgent:
    """Retrieves the highest-scoring evidence for a user query."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or HybridRetriever()

    def retrieve(self, query: str, evidence: Iterable[Evidence], top_k: int = 5) -> list[Evidence]:
        return self.retriever.search(query, evidence, top_k=top_k)
