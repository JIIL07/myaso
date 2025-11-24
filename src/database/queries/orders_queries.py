"""SQL запросы для работы с заказами."""

from typing import Any, Dict, List, Optional

from src.config.database_constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CREATED_AT,
    TABLE_ORDERS,
)
from src.utils import get_supabase_client


async def get_client_orders(phone: str) -> List[Dict[str, Any]]:
    """Получает заказы клиента по номеру телефона.

    Args:
        phone: Номер телефона клиента

    Returns:
        Список словарей с данными заказов
    """
    try:
        supabase = await get_supabase_client()
        result = (
            await supabase.table(TABLE_ORDERS)
            .select("*")
            .eq(COLUMN_CLIENT_PHONE, phone)
            .order(COLUMN_CREATED_AT, desc=True)
            .execute()
        )
        return result.data if result.data else []
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении заказов: {e}") from e


async def get_last_order(phone: str) -> Optional[Dict[str, Any]]:
    """Получает последний заказ клиента.

    Args:
        phone: Номер телефона клиента

    Returns:
        Словарь с данными последнего заказа или None
    """
    orders = await get_client_orders(phone)
    if orders:
        return orders[0]
    return None

