"""SQL запросы для работы с заказами."""

from typing import List, Optional

from src.config.database_constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CREATED_AT,
    TABLE_ORDERS,
)
from src.models.entities import Order
from src.utils.supabase_client import get_supabase_client


async def get_client_orders(phone: str) -> List[Order]:
    """Получает заказы клиента по номеру телефона.

    Args:
        phone: Номер телефона клиента

    Returns:
        Список моделей Order
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
        if result.data:
            return [Order(**order) for order in result.data]
        return []
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении заказов: {e}") from e


async def get_last_order(phone: str) -> Optional[Order]:
    """Получает последний заказ клиента.

    Args:
        phone: Номер телефона клиента

    Returns:
        Модель Order или None
    """
    orders = await get_client_orders(phone)
    if orders:
        return orders[0]
    return None

