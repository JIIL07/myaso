"""Инструменты для работы с клиентами."""

from __future__ import annotations

import logging
from langchain_core.tools import tool

from src.database.queries.clients_queries import get_client_profile_text
from src.database.queries.orders_queries import get_client_orders as get_client_orders_from_db

logger = logging.getLogger(__name__)


@tool
async def get_client_profile(phone: str) -> str:
    """Получает профиль клиента из базы данных.

    Возвращает информацию о клиенте: имя, контакты, город, бизнес-данные и т.д.

    Используй этот инструмент когда:
    - Нужна информация о клиенте для персонализации ответов
    - Нужно узнать город, бизнес-область или другие данные клиента
    - Нужно адаптировать предложения под профиль клиента

    Args:
        phone: Номер телефона клиента

    Returns:
        Строка с отформатированной информацией о профиле клиента или сообщение об отсутствии данных
    """
    try:
        return await get_client_profile_text(phone)
    except Exception as e:
        logger.error(f"Ошибка при получении профиля клиента: {e}", exc_info=True)
        return "Профиль клиента не найден в базе данных."


@tool
async def get_client_orders(phone: str) -> str:
    """Получает заказы клиента из базы данных.

    Возвращает список заказов клиента с информацией о товарах, ценах, датах доставки.

    Args:
        phone: Номер телефона клиента

    Returns:
        Строка с отформатированным списком заказов или сообщение об отсутствии заказов
    """
    try:
        orders = await get_client_orders_from_db(phone)

        if not orders:
            return "Заказы не найдены."

        orders_list = []
        for order in orders:
            order_info = [
                f"Товар: {order.get('title', 'Не указано')}",
                f"Дата: {order.get('created_at', 'Не указано')}",
                f"Вес (кг): {order.get('weight_kg', 'Не указано')}",
                f"Цена: {order.get('price_out', 'Не указано')}",
                f"Пункт назначения: {order.get('destination', 'Не указано')}",
            ]
            orders_list.append(
                "\n".join([info for info in order_info if "Не указано" not in info])
            )

        result_text = "\n\n---\n\n".join(orders_list)
        return f"Найдено заказов: {len(orders)}\n\n{result_text}"
    except Exception as e:
        logger.error(f"Ошибка при получении заказов: {e}", exc_info=True)
        return "Заказы не найдены."

