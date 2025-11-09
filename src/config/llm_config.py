"""Настройки LLM моделей."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class OpenRouterSettings(BaseSettings):
    """Настройки для OpenRouter API."""

    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    openrouter_api_key: str
    model_id: str
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class AlibabaSettings(BaseSettings):
    """Настройки для Alibaba DashScope API (embeddings)."""

    base_alibaba_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    alibaba_key: str
    embedding_model_id: str = "text-embedding-v4"
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.base_alibaba_url or self.base_alibaba_url == "":
            self.base_alibaba_url = (
                "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )

        if not self.embedding_model_id or self.embedding_model_id == "":
            self.embedding_model_id = "text-embedding-v4"

