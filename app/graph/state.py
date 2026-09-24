from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.models import Evidence


class IngestedFile(BaseModel):
    """An uploaded file, ingested once when it is uploaded and reused for every chat turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_id: str
    name: str
    size_bytes: int
    path: str
    status: str
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def is_queryable(self) -> bool:
        return self.status == "parsed"


class SessionState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = Field(..., description="Unique per-session identifier")
    uploaded_files: list[str] = Field(default_factory=list)
    files: dict[str, IngestedFile] = Field(default_factory=dict, description="Ingested files keyed by upload id")
    conversation_messages: list[dict[str, Any]] = Field(default_factory=list)
    active_context: str = ""
    last_question: str = ""
    last_answer: str = ""
    # Conversation memory (session-only): a running summary of turns older than the recent-history window.
    summary: str = ""
    summarized_upto: int = 0  # index into conversation_messages up to which turns are folded into ``summary``

    def queryable_evidence(self) -> list[Evidence]:
        return [item for file in self.files.values() if file.is_queryable for item in file.evidence]
