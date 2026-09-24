from __future__ import annotations

from typing import Iterable

from app.ingestion.models import Evidence
from app.retrieval.hybrid import HybridRetriever

# Spreadsheet payloads are answered by the data agent; embedding whole CSV sheets as text chunks only adds noise.
_NON_TEXT_SOURCES = {"table", "error", "empty", "unknown"}


class RetrievalAgent:
    """Retrieves the highest-scoring evidence for a user query."""

    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self._retriever = retriever

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def retrieve(
        self,
        query: str,
        evidence: Iterable[Evidence],
        top_k: int = 5,
        session_id: str | None = None,
        text_only: bool = False,
    ) -> list[Evidence]:
        # Placeholders ("OCR unavailable…") record that a file exists but carry no content to answer from.
        docs = [doc for doc in evidence if not doc.metadata.get("placeholder")]
        if text_only:
            docs = [doc for doc in docs if doc.source_type not in _NON_TEXT_SOURCES] or docs
        return self.retriever.search(query, docs, top_k=top_k, session_id=session_id)
