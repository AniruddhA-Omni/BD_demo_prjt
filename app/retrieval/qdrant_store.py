from __future__ import annotations

from typing import Any, Iterable

from app.ingestion.models import Evidence


class QdrantVectorStore:
    """Qdrant-backed vector store used as the dense retrieval layer for local document search."""

    def __init__(
        self,
        *,
        client: Any | None = None,
        collection_name: str = "document_intelligence",
        vector_size: int = 384,
        enabled: bool = False,
        url: str | None = None,
        timeout: float | int = 10,
        api_key: str | None = None,
        cloud_inference: bool = False,
    ) -> None:
        self.client = client
        if self.client is None and enabled:
            try:
                from qdrant_client import QdrantClient

                client_kwargs: dict[str, Any] = {
                    "url": url or "http://localhost:6333",
                    "timeout": timeout,
                }
                if api_key:
                    client_kwargs["api_key"] = api_key
                if cloud_inference:
                    client_kwargs["cloud_inference"] = True

                self.client = QdrantClient(**client_kwargs)
            except Exception:
                self.client = None
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.enabled = enabled and self.client is not None
        self._documents: list[Evidence] = []
        self._model: Any | None = None

    def add_documents(self, documents: Iterable[Evidence]) -> None:
        docs = list(documents)
        self._documents = docs
        if not self.enabled or self.client is None:
            return

        try:
            self._ensure_collection()
            points = []
            for doc in docs:
                embedding = self._embed(doc.content)
                if embedding is None:
                    continue
                points.append({
                    "id": str(doc.document_id),
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
            pass

    def search(self, query: str, evidence: Iterable[Evidence] | None = None, top_k: int = 5) -> list[Evidence]:
        docs = list(evidence) if evidence is not None else list(self._documents)
        if not docs:
            return []

        if not self.enabled or self.client is None:
            return docs[:top_k]

        try:
            self.add_documents(docs)
            self._ensure_collection()
            vector = self._embed(query)
            if vector is None:
                return docs[:top_k]
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                with_payload=True,
            )
            matches: list[Evidence] = []
            for point in results.points if hasattr(results, "points") else []:
                payload = getattr(point, "payload", {}) or {}
                metadata = dict(payload.get("metadata") or {})
                score = getattr(point, "score", None)
                if score is None:
                    score = payload.get("score")
                if score is not None:
                    metadata["dense_score"] = float(score)
                evidence = Evidence(
                    document_id=str(payload.get("document_id") or point.id),
                    file_name=str(payload.get("file_name") or "unknown.txt"),
                    file_type=str(payload.get("file_type") or "txt"),
                    source_type=str(payload.get("source_type") or "text"),
                    content=str(payload.get("content") or ""),
                    metadata=metadata,
                ) # type: ignore
                matches.append(evidence)
            return matches or docs[:top_k]
        except Exception:
            return docs[:top_k]

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

        if self._model is None:
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        vector = self._model.encode([text], normalize_embeddings=True)
        return vector[0].tolist()
