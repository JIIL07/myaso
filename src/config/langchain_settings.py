from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Load .env file explicitly
load_dotenv()


class LangChainSettings(BaseSettings):
    temperature: float = 0.7
    max_tokens: int = 2048
    model_name: str = "gpt-4o-mini"
    langsmith_api_key: str = ""
    langsmith_project_name: str = "myaso-agents"
    langsmith_tracing_enabled: bool = False

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def setup_langsmith_tracing(self) -> None:
        """Настраивает переменные окружения для LangSmith трейсинга."""
        if self.langsmith_tracing_enabled and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project_name
        else:
            # Отключаем трейсинг, если не настроен
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
            os.environ.pop("LANGCHAIN_API_KEY", None)
            os.environ.pop("LANGCHAIN_PROJECT", None)

