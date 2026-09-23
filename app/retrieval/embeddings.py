from __future__ import annotations

from typing import Iterable


class LocalEmbeddingModel:
    """Small local embedding wrapper with a graceful fallback when no model is available."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            return None

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float] | None:
        model = self._load_model()
        if model is None:
            return None
        vector = model.encode([text], normalize_embeddings=True)
        return vector[0].tolist()

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []

        model = self._load_model()
        if model is None:
            return []

        vectors = model.encode(list(values), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
