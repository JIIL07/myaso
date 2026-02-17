"""SQL запросы для работы с контекстом агента."""

import logging
from typing import Any, Dict, Optional

from src.services.database.constants import TABLE_AGENT_CONTEXT
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout

logger = logging.getLogger(__name__)


async def get_agent_context_from_db(client_phone: str) -> Optional[Dict[str, Any]]:
    """Получает контекст агента из базы данных.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Словарь с контекстом агента или None если не найден
        
    Raises:
        RuntimeError: Если произошла ошибка при выполнении запроса или timeout
    """
    try:
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table(TABLE_AGENT_CONTEXT)
            .select("context_data")
            .eq("client_phone", client_phone)
            .limit(1)
            .execute(),
            operation_name=f"get_agent_context_from_db({client_phone})",
        )
        
        if result.data and len(result.data) > 0:
            return result.data[0].get("context_data")
        return None
    except Exception as e:
        logger.error(f"Ошибка при получении контекста агента для {client_phone}: {e}")
        return None


async def save_agent_context_to_db(
    client_phone: str, context: Dict[str, Any]
) -> bool:
    """Сохраняет контекст агента в базу данных.

    Args:
        client_phone: Номер телефона клиента
        context: Словарь с контекстом агента

    Returns:
        True если контекст успешно сохранен, False в случае ошибки
        
    Raises:
        RuntimeError: Если произошла ошибка при выполнении запроса или timeout
    """
    try:
        supabase = await get_supabase_client()
        await execute_with_timeout(
            supabase.table(TABLE_AGENT_CONTEXT)
            .upsert(
                {
                    "client_phone": client_phone,
                    "context_data": context,
                },
                on_conflict="client_phone",
            )
            .execute(),
            operation_name=f"save_agent_context_to_db({client_phone})",
        )
        return True
    except Exception as e:
        logger.error(f"Ошибка при сохранении контекста агента для {client_phone}: {e}")
        return False
