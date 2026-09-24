from __future__ import annotations

import logging
import time
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class LocalEmbeddingModel:
    """Lazy sentence-transformers wrapper. Loading failures are recorded in ``last_error`` instead of raising."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._load_failed = False
        self.last_error = ""

    @property
    def dimension(self) -> int | None:
        model = self._load_model()
        if model is None:
            return None
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())

    def available(self) -> bool:
        return self._load_model() is not None

    def _load_model(self) -> Any | None:
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                "Embedding model loaded: %s (dimension=%s) in %.1f s", self.model_name, self.dimension, time.perf_counter() - started
            )
        except Exception as exc:
            self._load_failed = True
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Embedding model %s unavailable; dense retrieval disabled: %s", self.model_name, exc)
        return self._model

    def embed(self, text: str) -> list[float] | None:
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else None

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        model = self._load_model()
        if not values or model is None:
            return []
        vectors = model.encode(values, normalize_embeddings=True, batch_size=32)
        return [vector.tolist() for vector in vectors]
