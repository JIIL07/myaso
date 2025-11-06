"""Настройки базы данных."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class SupabaseSettings(BaseSettings):
    """Настройки для Supabase."""

    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

