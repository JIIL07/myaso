import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class LangChainSettings(BaseSettings):
    temperature: float = 0.8
    model_name: str = "gpt-4o-mini"
    langsmith_api_key: str = ""
    langsmith_project_name: str = "myaso-agents"
    langsmith_tracing_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def setup_langsmith_tracing(self) -> None:
        """Настраивает переменные окружения для LangSmith трейсинга.

        Если langsmith_tracing_enabled=True и указан API ключ, устанавливает
        переменные окружения для автоматического трейсинга всех LangChain операций.
        Если трейсинг отключен, очищает переменные окружения.
        """
        if not self.langsmith_tracing_enabled:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)
            return

        if not self.langsmith_api_key:
            return

        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project_name
