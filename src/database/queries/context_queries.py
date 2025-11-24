from typing import Any, Dict, Optional

from src.config.database_constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CONTEXT_DATA,
    TABLE_AGENT_CONTEXT,
)
from src.models.entities import AgentContext
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


async def get_agent_context_model(client_phone: str) -> Optional[AgentContext]:
    """Получает полную модель контекста агента из базы данных.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Модель AgentContext или None если не найден
    """
    try:
        supabase = await get_supabase_client()
        result = (
            await supabase.table(TABLE_AGENT_CONTEXT)
            .select("*")
            .eq(COLUMN_CLIENT_PHONE, client_phone)
            .execute()
        )
        if result.data and len(result.data) > 0:
            return AgentContext(**result.data[0])
        return None
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


async def save_agent_context_model(context: AgentContext) -> None:
    """Сохраняет модель контекста агента в базу данных.

    Args:
        context: Модель AgentContext для сохранения
    """
    try:
        supabase = await get_supabase_client()
        await (
            supabase.table(TABLE_AGENT_CONTEXT)
            .upsert(
                context.model_dump(exclude_none=True),
                on_conflict=COLUMN_CLIENT_PHONE,
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
            supabase.table(TABLE_AGENT_CONTEXT)
            .delete()
            .eq(COLUMN_CLIENT_PHONE, client_phone)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Ошибка при очистке контекста агента: {e}") from e

