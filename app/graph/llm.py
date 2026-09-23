from __future__ import annotations

import os

from langchain_ollama import ChatOllama

from app.config import Settings


def build_llm() -> ChatOllama:
    settings = Settings()

    llm_model = os.getenv("LLM_MODEL", settings.ollama_llm_model or "deepseek-r1:1.5b")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", str(settings.llm_temperature)))

    print(f"Initializing LLM model {llm_model} with temperature {llm_temperature}")
    return ChatOllama(model=llm_model, temperature=llm_temperature)
