import os

from app.config import Settings


def test_settings_include_ollama_model_and_temperature(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)

    settings = Settings()

    assert settings.llm_model == "gemma4:e4b "
    assert settings.llm_temperature == 0.0



def test_build_llm_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemma4:e4b ")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")

    from app.graph.llm import build_llm

    llm = build_llm()

    assert llm.model == "llama3.1:8b"
    assert llm.temperature == 0.2
