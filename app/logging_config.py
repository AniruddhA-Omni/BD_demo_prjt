"""Application logging.

Everything logs under the ``app`` logger hierarchy (``logging.getLogger(__name__)`` in each module). This module
configures only that hierarchy: Streamlit's and third-party loggers keep their own setup, and noisy libraries are capped
at WARNING.

Every record carries the current session and request id (``[session=… request=…]``), set through ``log_context`` and
stored in contextvars, so all lines for one question can be grepped together. LangGraph copies the context into its
parallel node threads, so the ids follow the parallel agents too.

Questions, answers and document text are never logged unless ``LOG_CONTENT=true``: use ``content()`` for any
user- or document-derived text.
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import sys
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import Settings, load_settings

APP_LOGGER = "app"
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "sentence_transformers", "qdrant_client", "langsmith", "transformers")
_HANDLER_MARK = "_bd_app_handler"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [session=%(session_id)s request=%(request_id)s] %(message)s"

_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_log_content = False


class ContextFilter(logging.Filter):
    """Stamps the current session/request id on every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.session_id = _session_id.get()
        record.request_id = _request_id.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "session": getattr(record, "session_id", "-"),
            "request": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that never fails on characters the console can't encode (Windows cp1252 consoles)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            self.stream.write(message.encode(encoding, "backslashreplace").decode(encoding) + self.terminator)
            self.flush()
        except Exception:  # pragma: no cover - logging must never break the app
            self.handleError(record)


class RingBufferHandler(logging.Handler):
    """Keeps the most recent records in memory for the UI's debug log panel."""

    def __init__(self, capacity: int = 200) -> None:
        super().__init__()
        self.records: deque[tuple[str, str]] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append((getattr(record, "session_id", "-"), self.format(record)))
        except Exception:  # pragma: no cover - logging must never break the app
            self.handleError(record)


_ring_buffer = RingBufferHandler()


def configure_logging(settings: Settings | None = None) -> None:
    """Configure the ``app`` logger once per process; later calls (e.g. every Streamlit rerun) are no-ops."""
    global _log_content
    settings = settings or load_settings()
    app_logger = logging.getLogger(APP_LOGGER)
    _log_content = settings.log_content
    if any(getattr(handler, _HANDLER_MARK, False) for handler in app_logger.handlers):
        return

    formatter: logging.Formatter = JsonFormatter() if settings.log_format == "json" else logging.Formatter(_TEXT_FORMAT)
    handlers: list[logging.Handler] = [SafeStreamHandler(sys.stderr), _ring_buffer]
    if settings.log_file:
        path = Path(settings.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"))

    context_filter = ContextFilter()
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(context_filter)
        setattr(handler, _HANDLER_MARK, True)
        app_logger.addHandler(handler)

    app_logger.setLevel(settings.log_level.upper())
    # Our handlers are attached here, so don't also hand records to the root logger (avoids duplicate lines when a
    # library configures root logging).
    app_logger.propagate = False
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def reset_logging() -> None:
    """Remove the handlers added by ``configure_logging`` (used by tests)."""
    global _log_content
    app_logger = logging.getLogger(APP_LOGGER)
    for handler in list(app_logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            app_logger.removeHandler(handler)
            if handler is not _ring_buffer:
                handler.close()
    app_logger.propagate = True
    app_logger.setLevel(logging.NOTSET)
    _ring_buffer.records.clear()
    _log_content = False


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


@contextmanager
def log_context(*, session_id: str | None = None, request_id: str | None = None) -> Iterator[None]:
    """Set the session/request id for records logged inside the block (and in threads that copy the context)."""
    tokens = []
    if session_id is not None:
        tokens.append((_session_id, _session_id.set(session_id)))
    if request_id is not None:
        tokens.append((_request_id, _request_id.set(request_id)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            try:
                var.reset(token)
            except ValueError:
                # A generator using this context (stream_query) was closed from another context; nothing to restore.
                pass


def set_session(session_id: str) -> None:
    """Bind the session id for the rest of the current context (one Streamlit script run)."""
    _session_id.set(session_id)


def current_request_id() -> str:
    return _request_id.get()


def content(text: Any, limit: int = 200) -> str:
    """Log-safe rendering of user/document text: the text itself only when LOG_CONTENT is on, else its length."""
    value = str(text)
    if not _log_content:
        return f"<{len(value)} chars>"
    return value if len(value) <= limit else value[: limit - 1] + "…"


def recent_logs(session_id: str | None = None) -> list[str]:
    """Most recent formatted log lines, optionally only those of one session (plus session-less startup lines)."""
    return [line for session, line in _ring_buffer.records if session_id is None or session in {session_id, "-"}]
