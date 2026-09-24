"""LangSmith wiring.

pydantic-settings reads `.env` into `Settings` but never exports it, and LangChain/LangGraph only look at the process
environment (`LANGSMITH_*`). `configure_tracing` bridges the two so LangGraph nodes, chat-model calls and the
`@traceable` steps below all land in one LangSmith trace.

With `LANGSMITH_HIDE_CONTENT=true`, run inputs and outputs (questions, document text, answers) are hidden from
LangSmith, and `record_step` attaches a content-free summary of each graph step (route, counts, scores, timings) as run
metadata so traces stay useful for debugging.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Mapping, TypeVar

from app.config import Settings, load_settings

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

_TRACING_ENV_KEYS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
_HIDE_ENV_KEYS = ("LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS")


def configure_tracing(settings: Settings | None = None) -> bool:
    """Enable LangSmith tracing when it is switched on and credentials exist; explicitly disable it otherwise."""
    settings = settings or load_settings()
    enabled = bool(settings.enable_langsmith and settings.langchain_tracing and settings.langchain_api_key)
    hide = enabled and settings.langsmith_hide_content

    for key in _TRACING_ENV_KEYS:
        os.environ[key] = "true" if enabled else "false"
    for key in _HIDE_ENV_KEYS:
        os.environ[key] = "true" if hide else "false"
    if enabled:
        os.environ["LANGSMITH_API_KEY"] = settings.langchain_api_key
        os.environ["LANGSMITH_ENDPOINT"] = settings.langchain_api_url
        os.environ["LANGSMITH_PROJECT"] = settings.langchain_project
    _clear_langsmith_env_cache()
    if enabled:
        logger.info("LangSmith tracing enabled (project=%s, hide_content=%s)", settings.langchain_project, hide)
    return enabled


def tracing_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() == "true"


def content_hidden() -> bool:
    return os.environ.get("LANGSMITH_HIDE_INPUTS", "").lower() == "true"


def traceable(name: str, run_type: str = "chain") -> Callable[[F], F]:
    """`langsmith.traceable` when available (a no-op while tracing is disabled), identity otherwise."""
    try:
        from langsmith import traceable as ls_traceable
    except Exception:  # pragma: no cover - langsmith ships with langchain-core
        return lambda func: func
    return ls_traceable(name=name, run_type=run_type)  # type: ignore[return-value]


def record_step(name: str, metadata: Mapping[str, Any]) -> None:
    """Attach a content-free step summary to the current trace as run metadata (no-op when tracing is off).

    Metadata is not affected by LANGSMITH_HIDE_INPUTS/OUTPUTS, so callers must only pass names, counts, scores and
    timings — never question, answer or document text.
    """
    if not tracing_enabled():
        return
    try:
        from langsmith.run_helpers import trace

        with trace(name=f"step:{name}", run_type="chain", inputs={}, metadata=dict(metadata)):
            pass
    except Exception as exc:  # pragma: no cover - tracing must never break a query
        logger.debug("Could not record step metadata for %s: %s", name, exc)


def _clear_langsmith_env_cache() -> None:
    """The LangSmith SDK caches environment lookups; clear it so the values just exported take effect."""
    try:
        from langsmith import utils as ls_utils

        ls_utils.get_env_var.cache_clear()
    except Exception:  # pragma: no cover
        pass
