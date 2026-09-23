from __future__ import annotations

from typing import Iterable

from rank_bm25 import BM25Okapi

from app.ingestion.models import Evidence


class BM25Retriever:
    """Keyword retrieval using BM25 over document content."""

    def search(self, query: str, evidence: Iterable[Evidence], top_k: int = 5) -> list[Evidence]:
        docs = list(evidence)
        if not docs:
            return []

        if not query or not query.strip():
            return docs[:top_k]

        tokenized_query = self._tokenize(query)
        tokenized_docs = [self._tokenize(doc.content) for doc in docs]

        if not any(tokenized_docs):
            return docs[:top_k]

        bm25 = BM25Okapi(tokenized_docs)
        base_scores = bm25.get_scores(tokenized_query)

        ranked = []
        for score, doc in zip(base_scores, docs):
            lexical_hits = sum(1 for token in tokenized_query if token in self._tokenize(doc.content))
            exact_term_bonus = 2.0 * lexical_hits
            if query.lower() in doc.content.lower():
                exact_term_bonus += 1.0
            ranked.append((score + exact_term_bonus, doc))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in ranked[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in text.replace("-", " ").split() if token.strip()]
