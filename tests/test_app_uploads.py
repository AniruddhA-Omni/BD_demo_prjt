from pathlib import Path

import app.main as main_module
from app.graph.graph import run_query
from app.graph.state import SessionState


class FakeUpload:
    def __init__(self, file_id: str, name: str, data: bytes) -> None:
        self.file_id, self.name, self._data = file_id, name, data
        self.size = len(data)

    def getvalue(self) -> bytes:
        return self._data


def test_uploads_are_saved_ingested_once_and_removed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path)
    ingested: list[str] = []
    original = main_module.ingest_upload

    def tracking_ingest(session, file_id, name, data):
        ingested.append(file_id)
        return original(session, file_id, name, data)

    monkeypatch.setattr(main_module, "ingest_upload", tracking_ingest)

    session = SessionState(session_id="s1")
    upload = FakeUpload("f1", "report.txt", b"Revenue increased by 12% in Q4.")

    main_module._sync_uploads(session, [upload], max_files=50)
    main_module._sync_uploads(session, [upload], max_files=50)

    assert ingested == ["f1"]
    saved = Path(session.files["f1"].path)
    assert saved.read_bytes() == b"Revenue increased by 12% in Q4."
    assert session.uploaded_files == ["report.txt"]

    result = run_query("What happened in Q4?", session.queryable_evidence())
    assert "12%" in result["answer"]

    main_module._sync_uploads(session, [], max_files=50)
    assert session.files == {}
    assert not saved.exists()


def test_upload_limits_are_enforced(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path)
    monkeypatch.setattr(main_module, "MAX_FILE_BYTES", 10)
    session = SessionState(session_id="s2")

    warnings = main_module._sync_uploads(
        session,
        [FakeUpload("a", "a.txt", b"small"), FakeUpload("b", "b.txt", b"x" * 11), FakeUpload("c", "c.txt", b"tiny")],
        max_files=1,
    )

    assert list(session.files) == ["a"]
    assert len(warnings) == 2


def test_failed_files_are_excluded_from_query_evidence(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path)
    session = SessionState(session_id="s3")

    main_module._sync_uploads(session, [FakeUpload("bad", "broken.pdf", b"%PDF-1.4 junk")], max_files=50)

    assert session.files["bad"].status != "parsed"
    assert session.queryable_evidence() == []
