from typing import Any, Dict, Optional

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
            await supabase.table("agent_context")
            .select("context_data")
            .eq("client_phone", client_phone)
            .execute()
        )
        if result.data and len(result.data) > 0:
            return result.data[0].get("context_data", {})
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
            supabase.table("agent_context")
            .upsert(
                {
                    "client_phone": client_phone,
                    "context_data": context_data,
                },
                on_conflict="client_phone",
            )
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при сохранении контекста агента: {e}") from e


async def update_agent_context(
    client_phone: str, updates: Dict[str, Any]
) -> None:
    """Частично обновляет контекст агента в базе данных.

    Args:
        client_phone: Номер телефона клиента
        updates: Словарь с обновлениями (будет объединен с существующим context_data)
    """
    try:
        # Получаем текущий контекст
        current_context = await get_agent_context_from_db(client_phone)
        # Объединяем с обновлениями
        updated_context = {**current_context, **updates}
        # Сохраняем
        await save_agent_context_to_db(client_phone, updated_context)
    except Exception as e:
        raise RuntimeError(f"Ошибка при обновлении контекста агента: {e}") from e


async def clear_agent_context_from_db(client_phone: str) -> None:
    """Очищает контекст агента в базе данных.

    Args:
        client_phone: Номер телефона клиента
    """
    try:
        supabase = await get_supabase_client()
        await (
            supabase.table("agent_context")
            .delete()
            .eq("client_phone", client_phone)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при очистке контекста агента: {e}") from e

