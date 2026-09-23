from __future__ import annotations

from typing import Any, Iterable

from app.ingestion.models import Evidence


class QdrantVectorStore:
    """Thin Qdrant-backed vector store wrapper with graceful fallback semantics."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        collection_name: str = "document_intelligence",
        vector_size: int = 384,
        enabled: bool = False,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.enabled = enabled
        self._fallback: list[Evidence] = []

    def add_documents(self, documents: Iterable[Evidence]) -> None:
        docs = list(documents)
        if not self.enabled or self.client is None:
            self._fallback = docs
            return

        try:
            self._ensure_collection()
            points = []
            for doc in docs:
                embedding = self._embed(doc.content)
                if embedding is None:
                    continue
                points.append({
                    "id": doc.document_id,
                    "vector": embedding,
                    "payload": {
                        "document_id": doc.document_id,
                        "file_name": doc.file_name,
                        "file_type": doc.file_type,
                        "source_type": doc.source_type,
                        "content": doc.content,
                        "metadata": doc.metadata,
                    },
                })
            if points:
                self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception:
            self._fallback = docs

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if not self.enabled or self.client is None:
            return self._fallback[:top_k]

        try:
            self._ensure_collection()
            vector = self._embed(query)
            if vector is None:
                return self._fallback[:top_k]
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            docs: list[Evidence] = []
            for point in results.points if hasattr(results, "points") else []:
                payload = getattr(point, "payload", {}) or {}
                doc = Evidence(
                    document_id=str(payload.get("document_id") or point.id),
                    file_name=str(payload.get("file_name") or "unknown.txt"),
                    file_type=str(payload.get("file_type") or "txt"),
                    source_type=str(payload.get("source_type") or "text"),
                    content=str(payload.get("content") or ""),
                    metadata=dict(payload.get("metadata") or {}),
                )
                docs.append(doc)
            return docs or self._fallback[:top_k]
        except Exception:
            return self._fallback[:top_k]

    def _ensure_collection(self) -> None:
        if self.client is None:
            return
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"size": self.vector_size, "distance": "Cosine"},
            )

    def _embed(self, text: str) -> list[float] | None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None

        model = getattr(self, "_model", None)
        if model is None:
            model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            self._model = model

        vector = model.encode([text], normalize_embeddings=True)
        return vector[0].tolist()
