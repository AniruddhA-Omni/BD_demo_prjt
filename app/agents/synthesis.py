from __future__ import annotations

from typing import Iterable

from app.ingestion.models import Evidence


class SynthesisAgent:
    """Builds grounded answers with inline citations from retrieved evidence."""

    def synthesize(self, question: str, evidence: Iterable[Evidence]) -> str:
        docs = list(evidence)
        if not docs:
            return "I couldn't find sufficient evidence in the uploaded files to answer this reliably."

        selected = docs[:3]
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
