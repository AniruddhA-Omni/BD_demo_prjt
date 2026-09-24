import json
import logging
from pathlib import Path

import pytest

import app.main as main_module
from app import services
from app.config import Settings
from app.graph.graph import run_query
from app.graph.llm import LLMService
from app.ingestion.models import Evidence
from app.ingestion.router import IngestionRouter
from app.logging_config import (
    APP_LOGGER,
    ContextFilter,
    JsonFormatter,
    configure_logging,
    content,
    log_context,
    recent_logs,
    reset_logging,
)

REPORT = Evidence(document_id="r", chunk_id="r-1", file_name="report.txt", file_type="txt", content="Revenue increased by 12% in Q4.")
TABLE = Evidence(
    document_id="t", chunk_id="t-0", file_name="revenue.csv", file_type="csv", source_type="table",
    content="region,revenue\nNorth,120\nWest,150\n",
)


@pytest.fixture
def app_logs():
    """Configured app logging; returns the records emitted, stamped with session/request ids like real handlers."""
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    configure_logging(Settings(log_level="DEBUG"))
    collector = Collector()
    collector.addFilter(ContextFilter())
    logger = logging.getLogger(APP_LOGGER)
    logger.addHandler(collector)
    yield records
    logger.removeHandler(collector)
    reset_logging()


def _messages(records):
    return [record.getMessage() for record in records]


def test_configure_logging_is_idempotent_across_reruns():
    try:
        configure_logging(Settings())
        count = len(logging.getLogger(APP_LOGGER).handlers)
        configure_logging(Settings())
        configure_logging(Settings())
        assert len(logging.getLogger(APP_LOGGER).handlers) == count
    finally:
        reset_logging()


def test_document_load_is_logged_with_chunk_counts(app_logs, tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("# Intro\nHello.\n\n# Pricing\nPlans start at 10.", encoding="utf-8")

    IngestionRouter().ingest_file(path)
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 junk")
    IngestionRouter().ingest_file(tmp_path / "broken.pdf")

    messages = _messages(app_logs)
    assert any(m.startswith("Document loaded successfully: notes.md (md): 2 chunk(s), 2 section(s)") for m in messages)
    assert any(r.levelno >= logging.WARNING and "broken.pdf" in r.getMessage() for r in app_logs)


def test_query_lines_share_one_request_id_including_parallel_agents(app_logs):
    with log_context(session_id="s1"):
        run_query("Compare the report revenue with the spreadsheet totals.", [REPORT, TABLE])

    query_records = [r for r in app_logs if r.name == "app.graph.graph" and "compiled" not in r.getMessage()]
    messages = _messages(query_records)
    assert messages[0].startswith("Question received")
    assert messages[-1].startswith("Answer completed: route=multi_agent outcome=pass")
    assert any(m.startswith("Step data_agent finished") for m in messages)
    assert any(m.startswith("Step retrieval_agent finished") for m in messages)
    request_ids = {record.request_id for record in query_records}
    assert len(request_ids) == 1 and "-" not in request_ids
    assert {record.session_id for record in query_records} == {"s1"}


def test_caller_supplied_ids_reach_parallel_node_threads(app_logs):
    with log_context(session_id="sess-9", request_id="req-9"):
        run_query("Compare the report revenue with the spreadsheet totals.", [REPORT, TABLE])

    steps = [r for r in app_logs if r.getMessage().startswith("Step ")]
    assert {"data_agent", "retrieval_agent"} <= {r.getMessage().split()[1] for r in steps}
    assert {(r.session_id, r.request_id) for r in steps} == {("sess-9", "req-9")}


def test_question_and_document_text_are_not_logged_by_default(app_logs):
    secret_question = "What happened to the confidential merger in Q4?"
    run_query(secret_question, [REPORT])

    text = "\n".join(_messages(app_logs))
    assert "confidential merger" not in text
    assert "Revenue increased by 12%" not in text
    assert f"<{len(secret_question)} chars>" in text


def test_content_is_logged_when_enabled():
    try:
        configure_logging(Settings(log_content=True))
        assert content("hello world") == "hello world"
    finally:
        reset_logging()
    assert content("hello world") == "<11 chars>"


def test_fallbacks_are_logged_as_warnings(app_logs):
    class Exploding:
        def invoke(self, *args, **kwargs):
            raise ConnectionError("ollama down")

    LLMService(chat_model=Exploding()).complete("system", "user")

    assert any(r.levelno == logging.WARNING and "ollama down" in r.getMessage() for r in app_logs)


def test_json_format_emits_one_object_per_line():
    record = logging.LogRecord("app.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.session_id, record.request_id = "s", "r"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "hello world" and payload["session"] == "s" and payload["request"] == "r"


def test_startup_logs_components_and_summary(app_logs):
    main_module.log_startup.cache_clear()
    main_module.log_startup()

    messages = _messages(app_logs)
    assert any(m.startswith("App started:") for m in messages)
    assert any(m.startswith("Component LLM:") for m in messages)
    assert any(m.startswith("Component Dense retrieval:") for m in messages)
    assert any(m.startswith("Startup complete in") for m in messages)
    main_module.log_startup.cache_clear()


def test_recent_logs_are_filtered_by_session(app_logs):
    logger = logging.getLogger("app.test")
    with log_context(session_id="a"):
        logger.info("line for a")
    with log_context(session_id="b"):
        logger.info("line for b")

    lines = recent_logs("a")
    assert any("line for a" in line for line in lines)
    assert not any("line for b" in line for line in lines)


def test_upload_and_removal_are_logged(app_logs, tmp_path: Path, monkeypatch):
    from app.graph.state import SessionState

    monkeypatch.setattr(main_module, "UPLOAD_ROOT", tmp_path)
    services.set_llm(LLMService(settings=Settings(enable_llm=False)))
    session = SessionState(session_id="s")

    class Upload:
        file_id, name, size = "f", "notes.txt", 12

        def getvalue(self):
            return b"Hello there."

    main_module._sync_uploads(session, [Upload()], max_files=5)
    main_module._sync_uploads(session, [], max_files=5)

    messages = _messages(app_logs)
    assert any(m.startswith("Upload received: notes.txt") for m in messages)
    assert any(m.startswith("File removed: notes.txt") for m in messages)
