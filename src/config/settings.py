"""Основные настройки приложения.

Собирает все настройки из отдельных конфигурационных файлов.
"""

from pydantic import BaseModel

from .database_config import SupabaseSettings
from .langfuse_config import LangFuseConfig
from .llm_config import AlibabaSettings, OpenRouterSettings
from .whatsapp_config import WhatsAppSettings


class Settings(BaseModel):
    """Главный класс настроек, объединяющий все конфигурации."""

    supabase: SupabaseSettings = SupabaseSettings()
    openrouter: OpenRouterSettings = OpenRouterSettings()
    alibaba: AlibabaSettings = AlibabaSettings()
    whatsapp: WhatsAppSettings = WhatsAppSettings()
    langfuse: LangFuseConfig = LangFuseConfig()


settings = Settings()
