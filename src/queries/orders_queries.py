"""SQL запросы для работы с заказами."""

from typing import List

from src.entities.product import Product


async def get_client_orders(client_phone: str) -> List[Product]:
    """Получает список заказов клиента.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Список заказов клиента
    """
    return []


async def get_last_order(client_phone: str) -> Product | None:
    """Получает последний заказ клиента.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Последний заказ клиента или None если заказов нет
    """
    return None
