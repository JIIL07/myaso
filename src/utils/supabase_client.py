"""Singleton для Supabase клиента.

Предоставляет переиспользуемый async Supabase клиент вместо создания
нового клиента для каждого запроса.
"""

import logging
from typing import Optional
from supabase import acreate_client, AClient, AsyncClientOptions

from src.config.settings import settings

logger = logging.getLogger(__name__)

_supabase_client: Optional[AClient] = None


async def get_supabase_client() -> AClient:
    """Получает или создает singleton Supabase async клиент.

    Клиент создается один раз при первом вызове и переиспользуется
    для всех последующих запросов.

    Returns:
        AClient: Async Supabase клиент для работы с БД

    Raises:
        RuntimeError: Если не удалось создать клиент
    """
    global _supabase_client

    if _supabase_client is None:
        try:
            _supabase_client = await acreate_client(
                settings.supabase.supabase_url,
                settings.supabase.supabase_service_key,
                options=AsyncClientOptions(schema="myaso"),
            )
        except Exception as e:
            logger.error(f"Ошибка при создании Supabase клиента: {e}", exc_info=True)
            raise RuntimeError(f"Не удалось создать Supabase клиент: {e}") from e

    return _supabase_client


async def close_supabase_client() -> None:
    """Закрывает Supabase клиент.

    Должно вызываться при завершении приложения для корректного
    закрытия соединения.
    """
    global _supabase_client

    if _supabase_client is not None:
        _supabase_client = None

