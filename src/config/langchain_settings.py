from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file explicitly
load_dotenv()


class LangChainSettings(BaseSettings):
    temperature: float = 0.7
    max_tokens: int = 2048
    model_name: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

