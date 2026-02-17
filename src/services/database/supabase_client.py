"""Singleton для Supabase клиента.

Предоставляет переиспользуемый async Supabase клиент вместо создания
нового клиента для каждого запроса.
"""

import asyncio
import logging
from typing import Optional

from supabase import AClient, AsyncClientOptions, acreate_client

logger = logging.getLogger(__name__)

_supabase_client: Optional[AClient] = None
_lock = asyncio.Lock()


async def get_supabase_client() -> AClient:
    """Получает или создает singleton Supabase async клиент.

    Клиент создается один раз при первом вызове и переиспользуется
    для всех последующих запросов. Использует asyncio.Lock для thread-safety
    при параллельных запросах.

    Returns:
        AClient: Async Supabase клиент для работы с БД

    Raises:
        RuntimeError: Если не удалось создать клиент
    """
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    async with _lock:
        # Double-check после получения блокировки
        if _supabase_client is None:
            from src.config.settings import settings
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
