from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "BD Demo Project"
    app_description: str = "Local-first multi-agent document intelligence showcase"
    ollama_base_url: str = "http://localhost:11434"
    enable_langsmith: bool = False


def get_settings() -> dict[str, str | bool]:
    settings = Settings()
    return {
        "app_name": settings.app_name,
        "app_description": settings.app_description,
        "ollama_base_url": settings.ollama_base_url,
        "enable_langsmith": settings.enable_langsmith,
    }
