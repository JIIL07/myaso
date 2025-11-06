"""SQL запросы для работы с историей диалогов."""

from typing import List, Dict, Any
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
            await supabase.table("conversation_history")
            .select("*")
            .eq("client_phone", phone)
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
            supabase.table("conversation_history")
            .delete()
            .eq("client_phone", phone)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при очистке истории: {e}") from e

