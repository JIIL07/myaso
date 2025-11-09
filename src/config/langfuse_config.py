"""Конфигурация LangFuse."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class LangFuseConfig(BaseSettings):
    """Конфигурация LangFuse"""

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True
    langfuse_flush_interval: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **kwargs):
        """Инициализация с явной загрузкой переменных окружения."""
        if "langfuse_public_key" not in kwargs:
            kwargs["langfuse_public_key"] = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        if "langfuse_secret_key" not in kwargs:
            kwargs["langfuse_secret_key"] = os.getenv("LANGFUSE_SECRET_KEY", "")
        if "langfuse_host" not in kwargs:
            kwargs["langfuse_host"] = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        if "langfuse_enabled" not in kwargs:
            enabled_str = os.getenv("LANGFUSE_ENABLED", "true")
            kwargs["langfuse_enabled"] = enabled_str.lower() in ("true", "1", "yes", "on")
        if "langfuse_flush_interval" not in kwargs:
            flush_interval = os.getenv("LANGFUSE_FLUSH_INTERVAL", "1")
            kwargs["langfuse_flush_interval"] = int(flush_interval) if flush_interval.isdigit() else 1
        super().__init__(**kwargs)

    @classmethod
    def is_configured(cls) -> bool:
        """Проверяет, настроен ли LangFuse"""
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        return bool(public_key and secret_key)

    @classmethod
    def validate(cls):
        """Валидирует конфигурацию"""
        if not cls.is_configured():
            raise ValueError(
                "LangFuse not configured. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
            )
