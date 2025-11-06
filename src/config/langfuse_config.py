"""Настройки LangFuse."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class LangFuseSettings(BaseSettings):
    """Настройки для интеграции с LangFuse."""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    langfuse_enabled: bool = True
    langfuse_flush_interval: int = 1

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

