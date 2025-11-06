from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class SupabaseSettings(BaseSettings):
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: str
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class OpenRouterSettings(BaseSettings):
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    openrouter_api_key: str
    model_id: str
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class AlibabaSettings(BaseSettings):
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


class WhatsAppSettings(BaseSettings):
    """Настройки для интеграции с WhatsApp API."""

    whatsapp_api_base_url: str = ""
    send_message_endpoint: str = "/send-message"
    send_image_endpoint: str = "/sendImage"

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
    def send_image_url(self) -> str:
        """Возвращает полный URL для отправки изображений."""
        return f"{self.api_base_url}{self.send_image_endpoint}"


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


class Settings(BaseModel):
    supabase: SupabaseSettings = SupabaseSettings()
    openrouter: OpenRouterSettings = OpenRouterSettings()
    alibaba: AlibabaSettings = AlibabaSettings()
    whatsapp: WhatsAppSettings = WhatsAppSettings()
    langfuse: LangFuseSettings = LangFuseSettings()


settings = Settings()
