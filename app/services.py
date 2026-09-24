"""Process-wide service registry.

Models (embedder, cross-encoder, LLM client) are expensive to load and the in-process Qdrant index must be shared by
every Streamlit rerun and session, so each service is created lazily once per process. Tests replace services with
fakes via the `set_*` functions and `reset()` restores the defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.graph.llm import LLMService
    from app.retrieval.embeddings import LocalEmbeddingModel
    from app.retrieval.qdrant_store import QdrantVectorStore
    from app.retrieval.reranker import Reranker

_instances: dict[str, Any] = {}


def reset() -> None:
    close = getattr(_instances.get("vector_store"), "close", None)
    _instances.clear()
    if callable(close):
        close()


def get_llm() -> "LLMService":
    if "llm" not in _instances:
        from app.graph.llm import LLMService

        _instances["llm"] = LLMService()
    return _instances["llm"]


def set_llm(service: "LLMService") -> None:
    _instances["llm"] = service


def get_embedder() -> "LocalEmbeddingModel":
    if "embedder" not in _instances:
        from app.config import load_settings
        from app.retrieval.embeddings import LocalEmbeddingModel

        _instances["embedder"] = LocalEmbeddingModel(load_settings().embedding_model)
    return _instances["embedder"]


def set_embedder(embedder: Any) -> None:
    _instances["embedder"] = embedder


def get_vector_store() -> "QdrantVectorStore":
    if "vector_store" not in _instances:
        from app.retrieval.qdrant_store import QdrantVectorStore

        _instances["vector_store"] = QdrantVectorStore.from_settings(embedder=get_embedder())
    return _instances["vector_store"]


def set_vector_store(store: Any) -> None:
    _instances["vector_store"] = store


def get_reranker() -> "Reranker":
    if "reranker" not in _instances:
        from app.config import load_settings
        from app.retrieval.reranker import CrossEncoderReranker, Reranker

        settings = load_settings()
        _instances["reranker"] = (
            CrossEncoderReranker(settings.reranker_model) if settings.reranker == "cross-encoder" else Reranker()
        )
    return _instances["reranker"]


def set_reranker(reranker: Any) -> None:
    _instances["reranker"] = reranker
