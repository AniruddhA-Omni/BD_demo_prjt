"""Test isolation: never depend on the developer's `.env`, a running Ollama, a Qdrant server or model downloads.

Environment variables take precedence over `.env` in pydantic-settings, so pinning them here makes every
`Settings()` deterministic. Tests that need the LLM, dense retrieval or the cross-encoder inject fakes explicitly.
"""

import os

import pytest

os.environ.update(
    {
        "APP_NAME": "BD Demo Project",
        "ENABLE_LLM": "false",
        "ENABLE_LANGSMITH": "false",
        "LANGCHAIN_TRACING": "false",
        "QDRANT_ENABLED": "false",
        "QDRANT_MODE": "memory",
        "RERANKER": "lexical",
    }
)


@pytest.fixture(autouse=True)
def _reset_cached_services():
    from app import services
    from app.config import load_settings
    from app.logging_config import reset_logging

    load_settings.cache_clear()
    services.reset()
    yield
    load_settings.cache_clear()
    services.reset()
    reset_logging()
