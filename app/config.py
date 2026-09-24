from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "BD Demo Project"
    app_description: str = "Local-first multi-agent document intelligence showcase"
    page_title: str = "Document Intelligence Demo"

    # LLM (Ollama). The LLM is optional: every agent falls back to deterministic behaviour when it is unavailable.
    enable_llm: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "gemma4:e4b"
    ollama_vision_model: str = "qwen3.5:0.8b"  # empty = reuse ollama_llm_model (Gemma 4 is multimodal)
    llm_model: str = "gemma4:e4b "
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 120.0
    llm_num_ctx: int = 8192

    max_files_per_session: int = 50
    # Ingest uploads on a background thread pool so the UI stays responsive (see app/ingestion/background.py).
    background_ingestion: bool = True
    ingest_workers: int = 2
    # Describe images with the vision model at upload (slow) instead of lazily on the first question about them.
    vision_at_ingestion: bool = False

    # LangSmith is the only component allowed to send data off the machine.
    enable_langsmith: bool = True
    langchain_api_key: str = ""
    langchain_api_url: str = "https://api.smith.langchain.com"
    langchain_tracing: bool = True
    langchain_project: str = "demo-prjt-observation"
    # Hide run inputs/outputs (questions, document text, answers) from LangSmith; step metadata stays content-free.
    langsmith_hide_content: bool = False

    # Dense retrieval. "memory" = in-process Qdrant (session-only, the default), "embedded" = on-disk local Qdrant,
    # "server" = a Qdrant service at qdrant_url (e.g. `docker run -p 6333:6333 qdrant/qdrant`).
    qdrant_enabled: bool = True
    qdrant_mode: Literal["memory", "embedded", "server"] = "memory"
    qdrant_path: str = ".qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "document_intelligence"
    qdrant_timeout_seconds: int = 10
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Reranking and hybrid scoring.
    reranker: Literal["cross-encoder", "lexical"] = "cross-encoder"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_dense_weight: float = 0.35
    hybrid_bm25_weight: float = 0.35
    hybrid_lexical_weight: float = 0.30
    retrieval_top_k: int = 4
    retrieval_candidate_k: int = 20

    # Verification: how many times an answer with unsupported claims may be regenerated.
    max_regenerations: int = 1

    # Greetings, questions about the assistant and general knowledge are answered directly (clearly labelled as not
    # grounded in the uploaded files). Document questions never fall back to model knowledge.
    allow_general_answers: bool = True
    general_history_turns: int = 6

    # Logging (see app/logging_config.py). Questions, answers and document text are only logged when log_content is on.
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"
    log_file: str = ""
    log_content: bool = False


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    """Process-wide settings. Tests clear this cache (see tests/conftest.py) after changing the environment."""
    return Settings()
