from __future__ import annotations

from typing import Iterable

from app.ingestion.models import Evidence


class LocalVectorStore:
    """Minimal vector store abstraction that keeps the retrieval layer usable without a live Qdrant service."""

    def __init__(self, embedding_model: object | None = None) -> None:
        self.embedding_model = embedding_model
        self._items: list[tuple[Evidence, list[float]]] = []

    def add_documents(self, documents: Iterable[Evidence]) -> None:
        for doc in documents:
            vector = None
            if self.embedding_model is not None:
                vector = self.embedding_model.embed(doc.content)
            self._items.append((doc, vector or self._lexical_vector(doc.content)))

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not self._items:
            return []

        if self.embedding_model is None:
            return self._lexical_search(query, top_k)

        query_vector = self.embedding_model.embed(query)
        if query_vector is None:
            return self._lexical_search(query, top_k)

        scored = []
        for doc, vector in self._items:
            similarity = self._cosine_similarity(query_vector, vector)
            scored.append((similarity, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _lexical_search(self, query: str, top_k: int) -> list[Evidence]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [doc for doc, _ in self._items[:top_k]]

        scored = []
        for doc, _ in self._items:
            doc_tokens = self._tokenize(doc.content)
            overlap = sum(1 for token in query_tokens if token in doc_tokens)
            score = overlap + (1 if query.lower() in doc.content.lower() else 0)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def _lexical_vector(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        return [float(tokens.count(token)) for token in sorted(set(tokens))]

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in text.replace("-", " ").split() if token.strip()]

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0

        dot = sum(a * b for a, b in zip(left, right))
        left_norm = sum(a * a for a in left) ** 0.5
        right_norm = sum(b * b for b in right) ** 0.5
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
