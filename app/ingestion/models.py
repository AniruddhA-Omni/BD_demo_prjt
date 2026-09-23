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
