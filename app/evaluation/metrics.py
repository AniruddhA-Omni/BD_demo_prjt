"""Ranking metrics used by the evaluation runner (``app/evaluation/runner.py``)."""

from __future__ import annotations


def recall_at_k(relevant_documents: list[str], retrieved_documents: list[str], k: int | None = None) -> float:
    if not relevant_documents:
        return 0.0
    limit = k if k is not None else len(retrieved_documents)
    retrieved_top = retrieved_documents[:limit]
    hits = sum(1 for doc in relevant_documents if doc in retrieved_top)
    return hits / len(relevant_documents)


def reciprocal_rank(relevant_documents: list[str], retrieved_documents: list[str]) -> float:
    if not relevant_documents:
        return 0.0
    for index, doc in enumerate(retrieved_documents, start=1):
        if doc in relevant_documents:
            return 1.0 / index
    return 0.0
