"""Сервис для загрузки конфигурации агента из БД и настройки LLM."""
import logging
from typing import Dict, Any

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.services.ai.constants import (
    DEFAULT_TEMPERATURE,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_EXECUTION_TIME,
    AGENT_RECURSION_LIMIT,
)

load_dotenv()

logger = logging.getLogger(__name__)


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


class AgentConfigService:
    """Сервис для загрузки и валидации конфигурации агента."""

    @staticmethod
    async def get_llm_config() -> Dict[str, Any]:
        """Загружает конфигурацию LLM.

        Returns:
            Словарь с параметрами LLM
        """
        config = {
            "temperature": DEFAULT_TEMPERATURE,
        }

        return config

    @staticmethod
    async def get_agent_limits() -> Dict[str, int]:
        """Загружает лимиты выполнения агента.

        Returns:
            Словарь с лимитами агента
        """
        limits = {
            "max_iterations": MAX_AGENT_ITERATIONS,
            "max_execution_time": MAX_AGENT_EXECUTION_TIME,
            "recursion_limit": AGENT_RECURSION_LIMIT,
        }

        return limits

    @staticmethod
    async def get_retry_config() -> Dict[str, Dict[str, Any]]:
        """Загружает конфигурацию retry для инструментов и модели.

        Returns:
            Словарь с конфигурацией retry для tool и model
        """
        config = {
            "tool": {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "initial_delay": 1.0,
            },
            "model": {
                "max_retries": 2,
                "backoff_factor": 2.0,
                "initial_delay": 1.0,
            },
        }

        return config
