from __future__ import annotations

from typing import Iterable

from app.agents.retrieval import RetrievalAgent
from app.ingestion.models import Evidence
from app.observability import traceable


class CodeAgent:
    """Code questions: retrieve the most relevant function/class chunks plus each file's module overview.

    The overview gives the model the file's structure (imports, functions, classes) so explanations and generated
    documentation describe how the retrieved symbols fit together.
    """

    def __init__(self, retrieval: RetrievalAgent | None = None) -> None:
        self.retrieval = retrieval or RetrievalAgent()

    @traceable(name="code_agent")
    def run(self, question: str, evidence: Iterable[Evidence], top_k: int = 5, session_id: str | None = None) -> list[Evidence]:
        docs = list(evidence)
        code_docs = [doc for doc in docs if doc.source_type == "code"]
        if not code_docs:
            return self.retrieval.retrieve(question, docs, top_k=top_k, session_id=session_id)

        ranked = self.retrieval.retrieve(question, code_docs, top_k=top_k, session_id=session_id)
        files = {doc.document_id for doc in ranked}
        overviews = [
            doc for doc in code_docs if doc.document_id in files and doc.metadata.get("symbol_kind") == "module"
        ]
        seen = {doc.key for doc in ranked}
        return [*[doc for doc in overviews if doc.key not in seen], *ranked]
