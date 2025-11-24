from typing import Any, Dict

from src.config.database_constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CONTEXT_DATA,
    TABLE_AGENT_CONTEXT,
)
from src.utils import get_supabase_client


async def get_agent_context_from_db(client_phone: str) -> Dict[str, Any]:
    """Получает контекст агента из базы данных.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Словарь с данными контекста (context_data) или пустой словарь если не найден
    """
    try:
        supabase = await get_supabase_client()
        result = (
            await supabase.table(TABLE_AGENT_CONTEXT)
            .select(COLUMN_CONTEXT_DATA)
            .eq(COLUMN_CLIENT_PHONE, client_phone)
            .execute()
        )
        if result.data and len(result.data) > 0:
            return result.data[0].get(COLUMN_CONTEXT_DATA, {})
        return {}
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении контекста агента: {e}") from e


async def save_agent_context_to_db(
    client_phone: str, context_data: Dict[str, Any]
) -> None:
    """Сохраняет контекст агента в базу данных.

    Args:
        client_phone: Номер телефона клиента
        context_data: Словарь с данными контекста
    """
    try:
        supabase = await get_supabase_client()
        await (
            supabase.table(TABLE_AGENT_CONTEXT)
            .upsert(
                {
                    COLUMN_CLIENT_PHONE: client_phone,
                    COLUMN_CONTEXT_DATA: context_data,
                },
                on_conflict=COLUMN_CLIENT_PHONE,
            )
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при сохранении контекста агента: {e}") from e


