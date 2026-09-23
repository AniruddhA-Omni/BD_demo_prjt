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

    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "gemma4:e4b"
    llm_model: str = "deepseek-r1:1.5b"
    llm_temperature: float = 0.0

    max_files_per_session: int = 50
    enable_langsmith: bool = True
    langchain_api_key: str = ""
    langchain_api_url: str = "https://api.smith.langchain.com"
    langchain_tracing: bool = True
    langchain_project: str = "demo-prjt-observation"

    qdrant_enabled: bool = True
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_cloud_inference: bool = False
    qdrant_collection: str = "document_intelligence"
    qdrant_timeout_seconds: int = 10


def get_settings() -> dict[str, str | bool | int]:
    settings = Settings()
    return {
        "app_name": settings.app_name,
        "app_description": settings.app_description,
        "page_title": settings.page_title,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_llm_model": settings.ollama_llm_model,
        "llm_model": settings.llm_model,
        "llm_temperature": settings.llm_temperature, # type: ignore
        "max_files_per_session": settings.max_files_per_session,
        "enable_langsmith": settings.enable_langsmith,
        "langchain_api_key": settings.langchain_api_key,
        "langchain_api_url": settings.langchain_api_url,
        "langchain_tracing": settings.langchain_tracing,
        "langchain_project": settings.langchain_project,
        "qdrant_enabled": settings.qdrant_enabled,
        "qdrant_url": settings.qdrant_url,
        "qdrant_api_key": settings.qdrant_api_key,
        "qdrant_cloud_inference": settings.qdrant_cloud_inference,
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_timeout_seconds": settings.qdrant_timeout_seconds,
    }
