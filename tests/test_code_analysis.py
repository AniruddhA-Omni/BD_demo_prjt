from pathlib import Path

from app.ingestion.router import IngestionRouter


def test_ingestion_router_extracts_python_code_structure(tmp_path: Path):
    code_path = tmp_path / "sample.py"
    code_path.write_text(
        "def greet(name):\n    return f'Hello {name}'\n\nclass Demo:\n    def __init__(self):\n        self.value = 1\n",
        encoding="utf-8",
    )

    evidence = IngestionRouter().ingest_file(code_path)

    assert evidence
    assert evidence[0].source_type == "code"
    assert "def greet" in evidence[0].content
    assert "class Demo" in evidence[0].content
