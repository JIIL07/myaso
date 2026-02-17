"""Централизованные настройки приложения."""

from dotenv import load_dotenv

from src.services.ai.config import AlibabaSettings, OpenRouterSettings
from src.services.database.config import SupabaseSettings
from src.services.langfuse.config import LangFuseConfig
from src.services.telegram.config import TelegramSettings
from src.services.whatsapp.config import WhatsAppSettings

load_dotenv()


class Settings:
    """Централизованный объект настроек приложения."""

    def __init__(self):
        self.supabase = SupabaseSettings()
        self.whatsapp = WhatsAppSettings()
        self.telegram = TelegramSettings()
        self.langfuse = LangFuseConfig()
        self.openrouter = OpenRouterSettings()
        self.alibaba = AlibabaSettings()


settings = Settings()
