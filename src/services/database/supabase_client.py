import asyncio
import logging
from typing import Optional

from supabase import AClient, AsyncClientOptions, acreate_client

logger = logging.getLogger(__name__)

_supabase_client: Optional[AClient] = None
_lock = asyncio.Lock()


async def get_supabase_client() -> AClient:
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    async with _lock:
        if _supabase_client is None:
            from src.config.settings import settings
            try:
                _supabase_client = await acreate_client(
                    settings.supabase.supabase_url,
                    settings.supabase.supabase_service_key,
                    options=AsyncClientOptions(schema="myaso"),
                )
            except Exception as e:
                logger.error("[Supabase] Ошибка создания клиента: %s", e, exc_info=True)
                raise RuntimeError(f"Не удалось создать Supabase клиент: {e}") from e

    return _supabase_client
