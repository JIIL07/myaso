"""SQL запросы для работы с историей диалогов."""

from typing import Any, Dict, List

from src.services.database.constants import (
    COLUMN_CLIENT_PHONE,
    TABLE_CONVERSATION_HISTORY,
)
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout


async def get_conversation_history_count(phone: str) -> int:
    """Получает количество сообщений в истории диалога.

    Args:
        phone: Номер телефона клиента

    Returns:
        Количество сообщений
        
    Raises:
        RuntimeError: Если произошла ошибка при выполнении запроса или timeout
    """
    try:
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table(TABLE_CONVERSATION_HISTORY)
            .select("id", count="exact")
            .eq(COLUMN_CLIENT_PHONE, phone)
            .execute(),
            operation_name=f"get_conversation_history_count({phone})",
        )
        return result.count if result.count else 0
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении истории: {e}") from e


async def clear_conversation_history(phone: str) -> None:
    """Очищает историю диалога для клиента.

    Args:
        phone: Номер телефона клиента
        
    Raises:
        RuntimeError: Если произошла ошибка при выполнении запроса или timeout
    """
    try:
        supabase = await get_supabase_client()
        await execute_with_timeout(
            supabase.table(TABLE_CONVERSATION_HISTORY)
            .delete()
            .eq(COLUMN_CLIENT_PHONE, phone)
            .execute(),
            operation_name=f"clear_conversation_history({phone})",
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при очистке истории: {e}") from e
