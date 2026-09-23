import os

from app.config import Settings, get_settings


def test_settings_include_ollama_model_and_temperature(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)

    settings = Settings()

    assert settings.llm_model == "deepseek-r1:1.5b"
    assert settings.llm_temperature == 0.0

    config = get_settings()
    assert config["llm_model"] == "deepseek-r1:1.5b"
    assert config["llm_temperature"] == 0.0


def test_build_llm_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    from app.graph.llm import build_llm

    llm = build_llm()

    assert llm.model == "llama3.1:8b"
    assert llm.temperature == 0.2
