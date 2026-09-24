import logging
import os
import threading
import time
from pathlib import Path

import pytest

import app.main as main_module
from app.config import Settings
from app.graph.graph import run_query, step_summary
from app.graph.state import SessionState
from app.ingestion.background import IngestionManager
from app.ingestion.models import Evidence
from app.logging_config import APP_LOGGER, ContextFilter, log_context
from app.observability import configure_tracing, content_hidden, record_step


def _wait(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.01)


class FakeUpload:
    def __init__(self, file_id: str, name: str, data: bytes) -> None:
        self.file_id, self.name, self._data, self.size = file_id, name, data, len(data)

    def getvalue(self) -> bytes:
        return self._data


# ---------------------------------------------------------------- 4.2 background ingestion


def test_submit_returns_immediately_and_results_are_collected_once():
    release = threading.Event()

    def slow_ingest(file_id, name, data):
        release.wait(5)
        return f"ingested:{name}"

    manager = IngestionManager(slow_ingest, max_workers=1)
    manager.submit("f1", "a.txt", b"x")
    manager.submit("f2", "b.txt", b"y")

    _wait(lambda: any(job.status == "processing" for job in manager.jobs()))
    assert {job.status for job in manager.pending()} == {"processing", "queued"}
    assert manager.collect() == []

    release.set()
    _wait(lambda: not manager.pending())
    assert sorted(manager.collect()) == ["ingested:a.txt", "ingested:b.txt"]
    assert manager.collect() == []  # handed out once
    manager.shutdown()


def test_cancelled_job_result_is_discarded_and_cleaned_up():
    release = threading.Event()
    discarded = []
    manager = IngestionManager(lambda f, n, d: (release.wait(5), "result")[1], max_workers=1, on_discard=discarded.append)

    manager.submit("f1", "a.txt", b"x")
    _wait(lambda: manager.jobs()[0].status == "processing")
    manager.cancel("f1")
    release.set()

    _wait(lambda: discarded == ["result"])
    assert manager.collect() == []
    assert not manager.has_job("f1")
    manager.shutdown()


def test_failed_ingestion_is_reported_not_raised():
    def broken(file_id, name, data):
        raise OSError("disk full")

    manager = IngestionManager(broken, max_workers=1)
    manager.submit("f1", "a.txt", b"x")

    _wait(lambda: manager.failed())
    assert manager.failed()[0].error == "OSError: disk full"
    assert manager.collect() == []
    manager.shutdown()


def test_worker_logs_keep_the_session_id():
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Collector()
    handler.addFilter(ContextFilter())
    logger = logging.getLogger(APP_LOGGER)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        manager = IngestionManager(lambda f, n, d: logging.getLogger("app.test").info("working on %s", n) or "ok")
        with log_context(session_id="sess-bg"):
            manager.submit("f1", "a.txt", b"x")
        _wait(lambda: manager.collect() == ["ok"] or not manager.pending())
        manager.shutdown()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)

    worker = [r for r in records if r.getMessage() == "working on a.txt"]
    assert worker and worker[0].session_id == "sess-bg"


def test_sync_uploads_queues_files_and_counts_pending_towards_the_limit(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path)
    release = threading.Event()
    session = SessionState(session_id="s")

    def ingest(file_id, name, data):
        release.wait(5)
        return main_module.ingest_upload(session, file_id, name, data)

    manager = IngestionManager(ingest, max_workers=2, on_discard=lambda f: main_module.discard_ingested(session, f))
    uploads = [FakeUpload("a", "a.txt", b"Revenue grew 12%."), FakeUpload("b", "b.txt", b"Policy needs approval.")]

    warnings = main_module._sync_uploads(session, uploads + [FakeUpload("c", "c.txt", b"x")], max_files=2, manager=manager)
    assert session.files == {}  # nothing ingested inline
    assert len(warnings) == 1 and "limit" in warnings[0]  # two pending files already fill the session
    assert [row["Status"] for row in main_module.file_rows(session, manager)] in (["queued", "queued"], ["processing", "processing"], ["processing", "queued"], ["queued", "processing"])

    main_module._sync_uploads(session, uploads, max_files=2, manager=manager)  # a rerun must not resubmit
    assert len(manager.jobs()) == 2

    release.set()
    _wait(lambda: not manager.pending())
    main_module._sync_uploads(session, uploads, max_files=2, manager=manager)
    assert set(session.files) == {"a", "b"}
    assert "12%" in run_query("What happened to revenue?", session.queryable_evidence())["answer"]
    manager.shutdown()


def test_removing_an_upload_while_it_is_ingesting_cancels_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path)
    release = threading.Event()
    session = SessionState(session_id="s")

    def ingest(file_id, name, data):
        release.wait(5)
        return main_module.ingest_upload(session, file_id, name, data)

    manager = IngestionManager(ingest, max_workers=1, on_discard=lambda f: main_module.discard_ingested(session, f))
    main_module._sync_uploads(session, [FakeUpload("a", "a.txt", b"hello")], max_files=5, manager=manager)
    _wait(lambda: manager.jobs()[0].status == "processing")

    main_module._sync_uploads(session, [], max_files=5, manager=manager)  # user removed the file
    release.set()
    _wait(lambda: not (tmp_path / "s" / "a").exists())

    main_module._sync_uploads(session, [], max_files=5, manager=manager)
    assert session.files == {}
    manager.shutdown()


# ---------------------------------------------------------------- 5.2 trace privacy


@pytest.fixture
def _restore_tracing_env():
    keys = ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS", "LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT"]
    saved = {key: os.environ.get(key) for key in keys}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    configure_tracing(Settings(enable_langsmith=False))


def test_hide_content_exports_langsmith_hide_variables_and_clears_the_sdk_cache(_restore_tracing_env):
    from langsmith import utils as ls_utils

    configure_tracing(Settings(enable_langsmith=True, langchain_tracing=True, langchain_api_key="k"))
    assert ls_utils.get_env_var("HIDE_INPUTS") == "false"  # cached lookup

    configure_tracing(Settings(enable_langsmith=True, langchain_tracing=True, langchain_api_key="k", langsmith_hide_content=True))

    assert os.environ["LANGSMITH_HIDE_INPUTS"] == os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    assert ls_utils.get_env_var("HIDE_INPUTS") == "true"  # the SDK sees the new value, not the cached one
    assert content_hidden()


def test_hide_content_is_off_when_tracing_is_off(_restore_tracing_env):
    configure_tracing(Settings(enable_langsmith=False, langsmith_hide_content=True))

    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "false"


def test_step_metadata_never_contains_answer_or_document_text():
    entry = {
        "node": "data_agent",
        "duration_ms": 5.0,
        "answer": "Secret merger value 900",
        "plan": {"operation": "sum", "column": "budget", "group_by": None, "method": "rules"},
        "retrieved": [{"file": "report.md", "hybrid": 0.4}],
        "unsupported": ["Profit rose 90%"],
    }

    summary = step_summary(entry)

    assert "Secret" not in str(summary) and "Profit rose" not in str(summary)
    assert summary["answer_chars"] == len(entry["answer"])
    assert summary["retrieved"] == 1 and summary["unsupported"] == 1
    assert summary["plan"] == "sum(budget) by=None via=rules"


def test_record_step_is_a_noop_without_tracing_and_traces_metadata_with_it(monkeypatch, _restore_tracing_env):
    calls = []

    class FakeTrace:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import langsmith.run_helpers

    monkeypatch.setattr(langsmith.run_helpers, "trace", FakeTrace)
    configure_tracing(Settings(enable_langsmith=False))
    record_step("route", {"route": "data"})
    assert calls == []

    os.environ["LANGSMITH_TRACING"] = "true"
    record_step("route", {"route": "data", "duration_ms": 3})
    assert calls == [{"name": "step:route", "run_type": "chain", "inputs": {}, "metadata": {"route": "data", "duration_ms": 3}}]
