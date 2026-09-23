from __future__ import annotations

from pathlib import Path
import uuid

from app.ingestion.models import Evidence


class IngestionRouter:
    """Simple router that normalizes file ingestion for the initial app scaffold."""

    _TEXT_TYPES = {"txt", "md"}
    _MARKDOWN_TYPES = {"md"}
    _TABLE_TYPES = {"csv", "xlsx"}
    _IMAGE_TYPES = {"png", "jpg", "jpeg"}
    _CODE_TYPES = {"py"}

    def ingest_file(self, file_path: str | Path) -> list[Evidence]:
        path = Path(file_path)
        file_type = path.suffix.lower().lstrip(".") or "unknown"
        doc_id = str(uuid.uuid4())

        if file_type in self._CODE_TYPES:
            return [self._parse_code_file(path, doc_id, file_type)]

        if file_type in self._TEXT_TYPES:
            return [self._parse_text_file(path, doc_id, file_type)]

        if file_type in self._TABLE_TYPES:
            return [self._parse_table_file(path, doc_id, file_type)]

        if file_type in self._IMAGE_TYPES:
            return [self._parse_image_file(path, doc_id, file_type)]

        if file_type in {"pdf", "docx", "pptx"}:
            return [
                Evidence(
                    document_id=doc_id,
                    file_name=path.name,
                    file_type=file_type,
                    source_type="placeholder",
                    content=f"{path.name} queued for parsing; this is a placeholder evidence item for later ingestion support.",
                    metadata=self._stamp_metadata(path, status="queued", source_type="placeholder"),
                )
            ]

        return [
            Evidence(
                document_id=doc_id,
                file_name=path.name,
                file_type=file_type,
                source_type="unknown",
                content=f"Unsupported file type '{file_type}' queued for future handling.",
                metadata=self._stamp_metadata(path, status="unsupported", source_type="unknown"),
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
            metadata=self._stamp_metadata(path, status="parsed", source_type=source_type),
        )

    def _parse_code_file(self, path: Path, doc_id: str, file_type: str) -> Evidence:
        import ast

        source = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(source)
            defined_functions = []
            defined_classes = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    defined_functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    defined_classes.append(node.name)
            summary = "\n".join(
                [
                    "Python code analysis summary:",
                    f"Functions: {', '.join(defined_functions) if defined_functions else 'none'}",
                    f"Classes: {', '.join(defined_classes) if defined_classes else 'none'}",
                    "",
                    source.strip() or "Empty file.",
                ]
            )
            metadata = self._stamp_metadata(path, status="parsed", source_type="code")
            metadata["functions"] = defined_functions
            metadata["classes"] = defined_classes
        except SyntaxError:
            summary = source.strip() or "Empty file."
            metadata = self._stamp_metadata(path, status="parsed", source_type="code")

        return Evidence(
            document_id=doc_id,
            file_name=path.name,
            file_type=file_type,
            source_type="code",
            content=summary,
            metadata=metadata,
        )

    def _parse_table_file(self, path: Path, doc_id: str, file_type: str) -> Evidence:
        if file_type == "csv":
            content = path.read_text(encoding="utf-8", errors="ignore")
        else:
            import pandas as pd

            df = pd.read_excel(path)
            content = df.to_csv(index=False)

        return Evidence(
            document_id=doc_id,
            file_name=path.name,
            file_type=file_type,
            source_type="table",
            content=content.strip() or "Empty table.",
            metadata=self._stamp_metadata(path, status="parsed", source_type="table"),
        )

    def _parse_image_file(self, path: Path, doc_id: str, file_type: str) -> Evidence:
        try:
            from PIL import Image
            import pytesseract
        except ImportError:
            content = (
                f"Image file '{path.name}' was ingested, but OCR support is unavailable in this environment. "
                "Install Tesseract and the OCR dependencies to extract text from images."
            )
            metadata = self._stamp_metadata(path, status="parsed", source_type="image")
            return Evidence(
                document_id=doc_id,
                file_name=path.name,
                file_type=file_type,
                source_type="image",
                content=content,
                metadata=metadata,
            )

        try:
            image = Image.open(path)
            extracted_text = pytesseract.image_to_string(image)
            content = extracted_text.strip() or (
                f"Image file '{path.name}' was processed, but no text was detected by OCR."
            )
        except Exception:
            content = (
                f"Image file '{path.name}' was ingested, but OCR failed in this environment. "
                "Install a working Tesseract binary or provide a higher-quality image."
            )

        return Evidence(
            document_id=doc_id,
            file_name=path.name,
            file_type=file_type,
            source_type="image",
            content=content,
            metadata=self._stamp_metadata(path, status="parsed", source_type="image"),
        )

    def _stamp_metadata(self, path: Path, *, status: str, source_type: str) -> dict[str, str]:
        return {
            "source_name": path.name,
            "source_path": str(path),
            "source_type": source_type,
            "path": str(path),
            "status": status,
        }
