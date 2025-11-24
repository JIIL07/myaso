"""SQL запросы для работы с историей диалогов."""

from typing import Any, Dict, List

from src.config.database_constants import (
    COLUMN_CLIENT_PHONE,
    TABLE_CONVERSATION_HISTORY,
)
from src.utils import get_supabase_client


async def get_conversation_history_count(phone: str) -> int:
    """Получает количество сообщений в истории диалога.

    Args:
        phone: Номер телефона клиента

    Returns:
        Количество сообщений
    """
    try:
        supabase = await get_supabase_client()
        result = (
            await supabase.table(TABLE_CONVERSATION_HISTORY)
            .select("*")
            .eq(COLUMN_CLIENT_PHONE, phone)
            .execute()
        )
        return len(result.data) if result.data else 0
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении истории: {e}") from e


async def clear_conversation_history(phone: str) -> None:
    """Очищает историю диалога для клиента.

    Args:
        phone: Номер телефона клиента
    """
    try:
        supabase = await get_supabase_client()
        await (
            supabase.table(TABLE_CONVERSATION_HISTORY)
            .delete()
            .eq(COLUMN_CLIENT_PHONE, phone)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при очистке истории: {e}") from e

