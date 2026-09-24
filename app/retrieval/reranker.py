from __future__ import annotations

import logging
import re
import time
from typing import Any, Iterable

from app.ingestion.models import Evidence

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "then", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your", "explain", "describe",
    "tell", "me", "about", "show", "give", "please",
}


def normalize_token(token: str) -> str:
    """Cheap plural folding ("risks" -> "risk", "totals" -> "total") applied to queries and documents alike."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [normalize_token(token) for token in re.findall(r"[a-z0-9%]+", text.lower())]


def query_terms(text: str) -> set[str]:
    return {normalize_token(token) for token in re.findall(r"[a-z0-9%]+", text.lower()) if token not in _STOPWORDS}


class Reranker:
    """Lexical fallback reranker: coverage of distinct query terms, then term frequency as a tie-breaker."""

    name = "lexical"

    def available(self) -> bool:
        return True

    def status(self) -> str:
        return "lexical"

    def rerank(self, query: str, evidence: Iterable[Evidence]) -> list[Evidence]:
        docs = list(evidence)
        terms = query_terms(query)
        if not docs or not terms:
            return docs

        scored: list[tuple[float, int, Evidence]] = []
        for position, doc in enumerate(docs):
            tokens = tokenize(doc.search_text)
            token_set = set(tokens)
            coverage = sum(1 for term in terms if term in token_set) / len(terms)
            frequency = sum(tokens.count(term) for term in terms) / max(len(tokens), 1)
            score = coverage + min(frequency, 0.5)
            doc.metadata["rerank_score"] = round(score, 4)
            scored.append((score, -position, doc))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [doc for _, _, doc in scored]


class CrossEncoderReranker(Reranker):
    """Local cross-encoder (default ``cross-encoder/ms-marco-MiniLM-L-6-v2``) with the lexical reranker as fallback."""

    name = "cross-encoder"

    def __init__(self, model_name: str, model: Any | None = None) -> None:
        self.model_name = model_name
        self._model = model
        self._load_failed = False
        self.last_error = ""

    def available(self) -> bool:
        return self._load() is not None

    def status(self) -> str:
        if self.available():
            return f"cross-encoder ({self.model_name})"
        return f"lexical fallback (cross-encoder unavailable: {self.last_error})"

    def rerank(self, query: str, evidence: Iterable[Evidence]) -> list[Evidence]:
        docs = list(evidence)
        model = self._load()
        if model is None or not docs or not query.strip():
            return super().rerank(query, docs)
        try:
            scores = model.predict([(query, doc.search_text) for doc in docs])
        except Exception as exc:
            logger.warning("Cross-encoder scoring failed, using lexical rerank: %s", exc)
            return super().rerank(query, docs)
        for doc, score in zip(docs, scores):
            doc.metadata["rerank_score"] = round(float(score), 4)
        order = sorted(range(len(docs)), key=lambda index: float(scores[index]), reverse=True)
        return [docs[index] for index in order]

    def _load(self) -> Any | None:
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            started = time.perf_counter()
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder reranker loaded: %s in %.1f s", self.model_name, time.perf_counter() - started)
        except Exception as exc:
            self._load_failed = True
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Cross-encoder %s unavailable: %s", self.model_name, exc)
        return self._model
