"""SQL запросы для работы с заказами."""

from typing import List, Dict, Any, Optional
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
            await supabase.table("orders")
            .select("*")
            .eq("client_phone", phone)
            .order("created_at", desc=True)
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

