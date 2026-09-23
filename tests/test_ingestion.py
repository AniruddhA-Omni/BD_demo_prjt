from pathlib import Path

from app.ingestion.router import IngestionRouter


def test_ingestion_router_extracts_text_documents(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Revenue increased by 12% in Q4.\n", encoding="utf-8")

    evidence = IngestionRouter().ingest_file(file_path)

    assert len(evidence) == 1
    assert evidence[0].file_name == "notes.txt"
    assert evidence[0].file_type == "txt"
    assert "Revenue increased by 12%" in evidence[0].content


def test_ingestion_router_handles_unsupported_file_types(tmp_path: Path):
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"%PDF-1.4 not-a-real-pdf")

    evidence = IngestionRouter().ingest_file(file_path)

    assert len(evidence) == 1
    assert evidence[0].file_name == "report.pdf"
    assert evidence[0].file_type == "pdf"
    assert "queued for parsing" in evidence[0].content.lower()
