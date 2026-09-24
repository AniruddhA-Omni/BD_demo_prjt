from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    document_id: str
    file_name: str
    file_type: str
    source_type: str = "text"
    page: int | None = None
    section: str | None = None
    sheet: str | None = None
    cell_range: str | None = None
    chunk_id: str | None = None
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        """Text used for matching: the section heading (e.g. "Risks", a slide title) carries meaning the body lacks,
        and an image's cached visual description (added lazily by the vision agent) makes the image findable."""
        text = self.content
        if self.section and self.section not in text:
            text = f"{self.section}\n{text}"
        description = self.metadata.get("vision_description")
        if description:
            text = f"{text}\n{description}"
        return text

    @property
    def key(self) -> str:
        """Unique retrieval identifier: one file can yield many chunks sharing a document_id."""
        return self.chunk_id or self.document_id
