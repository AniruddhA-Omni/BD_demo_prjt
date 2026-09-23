from __future__ import annotations

import re
from typing import Iterable

from app.ingestion.models import Evidence

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "had", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "then", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your"
}


class SynthesisAgent:
    """Builds grounded answers with inline citations from retrieved evidence."""

    def synthesize(self, question: str, evidence: Iterable[Evidence]) -> str:
        docs = list(evidence)
        if not docs:
            return "I couldn't find sufficient evidence in the uploaded files to answer this reliably."

        filtered = [doc for doc in docs if self._is_relevant(question, doc.content)]
        if not filtered:
            return "I found insufficient evidence in the uploaded files to answer this reliably."

        selected = filtered[:3]
        answer_parts: list[str] = []

        for doc in selected:
            content = doc.content.strip()
            if not content:
                continue

            source = f"{doc.file_name}"
            if doc.metadata.get("page") is not None:
                source += f" — page {doc.metadata['page']}"
            if doc.metadata.get("section") is not None:
                source += f" — {doc.metadata['section']}"

            answer_parts.append(f"{content} [source: {source}]")

        if not answer_parts:
            return "I couldn't find sufficient evidence in the uploaded files to answer this reliably."

        return " ".join(answer_parts)

    def _is_relevant(self, question: str, content: str) -> bool:
        question_terms = self._normalize_terms(question)
        content_terms = self._normalize_terms(content)

        if not question_terms:
            return True

        overlap = len(question_terms & content_terms)
        if overlap > 0:
            return True

        phrase_terms = {term for term in question_terms if term not in {"process", "policy", "report", "document"}}
        if not phrase_terms:
            return False

        return any(term in content.lower() for term in phrase_terms)

    def _normalize_terms(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9%]+", text.lower())
        return {token for token in tokens if len(token) > 1 and token not in _STOPWORDS}
