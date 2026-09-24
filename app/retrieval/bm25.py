from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from rank_bm25 import BM25Okapi

from app.ingestion.models import Evidence
from app.retrieval.reranker import query_terms, tokenize


@dataclass(frozen=True)
class CorpusIndex:
    """Tokenised chunks plus their BM25 index, built once per distinct set of chunks."""

    bm25: BM25Okapi | None
    token_sets: tuple[frozenset[str], ...]


class _CorpusCache:
    """Small thread-safe LRU of ``CorpusIndex`` objects.

    Keyed by each chunk's key *and* a hash of its text, so the same set of uploaded chunks hits the cache on every
    question while any content change (or a different subset, e.g. text-only or image-only evidence) builds a new
    entry. Parallel agents query concurrently, hence the lock.
    """

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._entries: OrderedDict[tuple, CorpusIndex] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, docs: list[Evidence]) -> CorpusIndex:
        key = tuple((doc.key, hash(doc.search_text)) for doc in docs)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return entry
        entry = self._build(docs)
        with self._lock:
            self.misses += 1
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
        return entry

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = 0

    @staticmethod
    def _build(docs: list[Evidence]) -> CorpusIndex:
        tokenized = [tokenize(doc.search_text) for doc in docs]
        bm25 = BM25Okapi([tokens or [""] for tokens in tokenized]) if any(tokenized) else None
        return CorpusIndex(bm25=bm25, token_sets=tuple(frozenset(tokens) for tokens in tokenized))


corpus_cache = _CorpusCache()


class BM25Retriever:
    """Keyword retrieval using BM25 over chunk content, with the index cached across questions."""

    def index(self, evidence: Iterable[Evidence]) -> CorpusIndex:
        return corpus_cache.get(list(evidence))

    def scores(self, query: str, evidence: Iterable[Evidence]) -> list[float]:
        """Raw BM25 scores (clamped at 0) in the same order as ``evidence``."""
        docs = list(evidence)
        terms = [token for token in tokenize(query) if token in query_terms(query)]
        if not docs or not terms:
            return [0.0] * len(docs)
        corpus = corpus_cache.get(docs)
        if corpus.bm25 is None:
            return [0.0] * len(docs)
        return [max(float(score), 0.0) for score in corpus.bm25.get_scores(terms)]

    def search(self, query: str, evidence: Iterable[Evidence], top_k: int = 5) -> list[Evidence]:
        docs = list(evidence)
        if not docs or not query.strip():
            return docs[:top_k]

        terms = query_terms(query)
        token_sets = corpus_cache.get(docs).token_sets
        ranked = []
        for score, doc, tokens in zip(self.scores(query, docs), docs, token_sets):
            # Distinct-term coverage keeps tiny corpora sensible, where BM25 idf collapses towards zero.
            coverage = sum(1 for term in terms if term in tokens)
            ranked.append((score + coverage, doc))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]
