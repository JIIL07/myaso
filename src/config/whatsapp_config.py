"""Настройки WhatsApp API."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class WhatsAppSettings(BaseSettings):
    """Настройки для интеграции с WhatsApp API."""

    whatsapp_api_base_url: str = ""
    send_message_endpoint: str = "/send-message"
    send_file_endpoint: str = "/sendFile"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def api_base_url(self) -> str:
        """Возвращает базовый URL API WhatsApp."""
        return self.whatsapp_api_base_url

    @property
    def send_message_url(self) -> str:
        """Возвращает полный URL для отправки сообщений."""
        return f"{self.api_base_url}{self.send_message_endpoint}"

    @property
    def send_file_url(self) -> str:
        """Возвращает полный URL для отправки файлов."""
        return f"{self.api_base_url}{self.send_file_endpoint}"

