from __future__ import annotations

from pathlib import Path
import uuid

from app.ingestion.models import Evidence


class IngestionRouter:
    """Simple router that normalizes file ingestion for the initial app scaffold."""

    _TEXT_TYPES = {"txt", "md", "py"}
    _MARKDOWN_TYPES = {"md"}

    def ingest_file(self, file_path: str | Path) -> list[Evidence]:
        path = Path(file_path)
        file_type = path.suffix.lower().lstrip(".") or "unknown"
        doc_id = str(uuid.uuid4())

        if file_type in self._TEXT_TYPES:
            return [self._parse_text_file(path, doc_id, file_type)]

        if file_type in {"pdf", "docx", "pptx", "csv", "xlsx", "png", "jpg", "jpeg"}:
            return [
                Evidence(
                    document_id=doc_id,
                    file_name=path.name,
                    file_type=file_type,
                    source_type="placeholder",
                    content=f"{path.name} queued for parsing; this is a placeholder evidence item for later ingestion support.",
                    metadata={"path": str(path), "status": "queued"},
                )
            ]

        return [
            Evidence(
                document_id=doc_id,
                file_name=path.name,
                file_type=file_type,
                source_type="unknown",
                content=f"Unsupported file type '{file_type}' queued for future handling.",
                metadata={"path": str(path), "status": "unsupported"},
            )
        ]

    def _parse_text_file(self, path: Path, doc_id: str, file_type: str) -> Evidence:
        text = path.read_text(encoding="utf-8", errors="ignore")
        source_type = "markdown" if file_type in self._MARKDOWN_TYPES else "text"
        return Evidence(
            document_id=doc_id,
            file_name=path.name,
            file_type=file_type,
            source_type=source_type,
            content=text.strip() or "Empty file.",
            metadata={"path": str(path), "status": "parsed"},
        )
