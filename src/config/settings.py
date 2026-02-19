from functools import lru_cache

from dotenv import load_dotenv

from src.services.ai.config import AlibabaSettings, OpenRouterSettings
from src.services.database.config import SupabaseSettings
from src.services.langfuse.config import LangFuseConfig
from src.services.telegram.config import TelegramSettings
from src.services.whatsapp.config import WhatsAppSettings

# Single load_dotenv call for the whole application
load_dotenv()


class Settings:
    """Aggregated application settings."""

    def __init__(self) -> None:
        self.supabase = SupabaseSettings()
        self.whatsapp = WhatsAppSettings()
        self.telegram = TelegramSettings()
        self.langfuse = LangFuseConfig()
        self.openrouter = OpenRouterSettings()
        self.alibaba = AlibabaSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached)."""
    return Settings()


# Module-level alias for backwards compatibility
settings = get_settings()
