from pydantic_settings import BaseSettings, SettingsConfigDict


class WhatsAppSettings(BaseSettings):

    whatsapp_api_base_url: str = ""
    send_message_endpoint: str = "/send-message"
    send_file_endpoint: str = "/sendFile"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def api_base_url(self) -> str:
        return self.whatsapp_api_base_url

    @property
    def send_message_url(self) -> str:
        return f"{self.api_base_url}{self.send_message_endpoint}"

    @property
    def send_file_url(self) -> str:
        return f"{self.api_base_url}{self.send_file_endpoint}"
