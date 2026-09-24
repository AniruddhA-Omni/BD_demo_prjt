from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Iterable

from app.config import Settings, load_settings
from app.ingestion.models import Evidence

logger = logging.getLogger(__name__)


def point_id(key: str) -> str:
    """Qdrant point ids must be UUIDs (or ints); ingestion chunk ids already are, anything else is mapped stably."""
    try:
        return str(uuid.UUID(key))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class QdrantVectorStore:
    """Dense retrieval layer backed by Qdrant.

    Chunks are embedded and upserted once (``index``) with ``session_id``/``document_id`` payload fields, and every
    search is filtered by session and, optionally, by document. Failures never raise into the query path: they are
    logged, recorded in ``last_error`` and surfaced through ``status()`` so a broken setup is visible in the UI
    instead of silently degrading ranking.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        embedder: Any | None = None,
        collection_name: str = "document_intelligence",
        enabled: bool = False,
        mode: str = "server",
        path: str = ".qdrant",
        url: str | None = None,
        timeout: float | int = 10,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.embedder = embedder
        self.mode = mode
        self.client = client
        self.last_error = ""
        self._indexed: dict[str, str] = {}  # evidence key -> document_id
        self._lock = threading.RLock()  # background ingestion workers and queries use the store concurrently
        self._collection_ready = False

        if self.client is None and enabled:
            try:
                from qdrant_client import QdrantClient

                if mode == "memory":
                    self.client = QdrantClient(location=":memory:")
                elif mode == "embedded":
                    self.client = QdrantClient(path=path)
                else:
                    client_kwargs: dict[str, Any] = {"url": url or "http://localhost:6333", "timeout": timeout}
                    if api_key:
                        client_kwargs["api_key"] = api_key
                    self.client = QdrantClient(**client_kwargs)
            except Exception as exc:
                self._record_error("connect", exc)
                self.client = None
        self.enabled = enabled and self.client is not None
        if self.enabled:
            location = {"memory": "in-process memory", "embedded": f"local path {path}"}.get(mode, url or "http://localhost:6333")
            logger.info("Qdrant connected: mode=%s (%s), collection=%s", mode, location, collection_name)
        if not enabled:
            self.last_error = "disabled in settings"

    @classmethod
    def from_settings(cls, embedder: Any | None = None, settings: Settings | None = None) -> "QdrantVectorStore":
        settings = settings or load_settings()
        return cls(
            embedder=embedder,
            collection_name=settings.qdrant_collection,
            enabled=settings.qdrant_enabled,
            mode=settings.qdrant_mode,
            path=settings.qdrant_path,
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout_seconds,
            api_key=settings.qdrant_api_key or None,
        )

    def ready(self) -> bool:
        return bool(self.enabled and self.client is not None and self.embedder is not None and self.embedder.available())

    def is_indexed(self, key: str) -> bool:
        return key in self._indexed

    def status(self) -> str:
        if not self.enabled:
            return f"off ({self.last_error or 'disabled'})"
        if self.embedder is None or not self.embedder.available():
            reason = getattr(self.embedder, "last_error", "") or "no embedding model"
            return f"unavailable (embeddings: {reason})"
        if self.last_error:
            return f"degraded ({self.last_error})"
        return f"ready ({len(self._indexed)} chunks indexed)"

    def index(self, documents: Iterable[Evidence], session_id: str | None = None) -> int:
        """Embed and upsert chunks that aren't indexed yet. Returns how many were added."""
        with self._lock:
            return self._index(list(documents), session_id)

    def _index(self, documents: list[Evidence], session_id: str | None) -> int:
        docs = [doc for doc in documents if doc.key not in self._indexed and doc.content.strip()]
        if not docs or not self.ready():
            return 0
        try:
            from qdrant_client import models

            vectors = self.embedder.embed_batch([doc.search_text for doc in docs])  # type: ignore[union-attr]
            if len(vectors) != len(docs):
                return 0
            self._ensure_collection(len(vectors[0]))
            points = [
                models.PointStruct(id=point_id(doc.key), vector=vector, payload=self._payload(doc, session_id))
                for doc, vector in zip(docs, vectors)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)  # type: ignore[union-attr]
        except Exception as exc:
            self._record_error("index", exc)
            return 0
        self._indexed.update({doc.key: doc.document_id for doc in docs})
        self.last_error = ""
        logger.debug("Indexed %d chunks in Qdrant (session=%s)", len(docs), session_id)
        return len(docs)

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        session_id: str | None = None,
        document_ids: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """Return ``{evidence_key: cosine_score}`` for the best matches inside the session/document filter."""
        with self._lock:
            return self._search(query, top_k=top_k, session_id=session_id, document_ids=document_ids)

    def _search(
        self, query: str, *, top_k: int, session_id: str | None, document_ids: Iterable[str] | None
    ) -> dict[str, float]:
        if not query.strip() or not self.ready() or not self._indexed:
            return {}
        try:
            from qdrant_client import models

            vector = self.embedder.embed(query)  # type: ignore[union-attr]
            if vector is None:
                return {}
            conditions: list[Any] = []
            if session_id:
                conditions.append(models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)))
            ids = sorted(set(document_ids or []))
            if ids:
                conditions.append(models.FieldCondition(key="document_id", match=models.MatchAny(any=ids)))
            result = self.client.query_points(  # type: ignore[union-attr]
                collection_name=self.collection_name,
                query=vector,
                limit=top_k,
                query_filter=models.Filter(must=conditions) if conditions else None,
                with_payload=["key"],
            )
        except Exception as exc:
            self._record_error("search", exc)
            return {}
        scores: dict[str, float] = {}
        for point in getattr(result, "points", []):
            key = (point.payload or {}).get("key") or str(point.id)
            scores[str(key)] = float(point.score)
        return scores

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            self._delete("document_id", document_id)
            self._indexed = {key: doc_id for key, doc_id in self._indexed.items() if doc_id != document_id}
        logger.debug("Deleted vectors for document %s", document_id)

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            self._delete("session_id", session_id)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _delete(self, field: str, value: str) -> None:
        if not self.enabled or self.client is None or not self._collection_ready:
            return
        try:
            from qdrant_client import models

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))])
                ),
            )
        except Exception as exc:
            self._record_error("delete", exc)

    def _ensure_collection(self, dimension: int) -> None:
        if self._collection_ready:
            return
        from qdrant_client import models

        if not self.client.collection_exists(self.collection_name):  # type: ignore[union-attr]
            self.client.create_collection(  # type: ignore[union-attr]
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            logger.info("Qdrant collection created: %s (dimension=%d)", self.collection_name, dimension)
            # Local (memory/embedded) Qdrant filters without payload indexes and warns if asked to create them.
            for field in (("session_id", "document_id") if self.mode == "server" else ()):
                try:
                    self.client.create_payload_index(  # type: ignore[union-attr]
                        collection_name=self.collection_name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception as exc:
                    logger.info("Could not create payload index on %s: %s", field, exc)
        self._collection_ready = True

    def _payload(self, doc: Evidence, session_id: str | None) -> dict[str, Any]:
        return {
            "key": doc.key,
            "session_id": session_id or doc.metadata.get("session_id"),
            "document_id": doc.document_id,
            "chunk_id": doc.chunk_id,
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "source_type": doc.source_type,
            "page": doc.page,
            "section": doc.section,
            "sheet": doc.sheet,
            "cell_range": doc.cell_range,
        }

    def _record_error(self, operation: str, exc: Exception) -> None:
        self.last_error = f"{operation} failed: {type(exc).__name__}: {exc}"
        logger.warning("Qdrant %s", self.last_error)
