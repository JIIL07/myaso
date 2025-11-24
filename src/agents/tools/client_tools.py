"""Инструменты для работы с клиентами."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.database.queries.clients_queries import get_client_profile_text
from src.database.queries.orders_queries import (
    get_client_orders as get_client_orders_from_db,
)

logger = logging.getLogger(__name__)


@tool
async def get_client_profile(phone: str) -> str:
    """Получает профиль клиента из базы данных.

    НАЗНАЧЕНИЕ: Получает профиль клиента из базы данных

    ИСПОЛЬЗУЙ КОГДА:
    - Нужна информация о клиенте для персонализации ответов
    - Нужно узнать город клиента для предложения товаров из его региона
    - Нужно узнать бизнес-область клиента для адаптации предложений
    - Нужно адаптировать ответы под профиль клиента

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Информация о клиенте не нужна для ответа
    - Запрос не требует персонализации

    Args:
        phone: Номер телефона клиента (в формате +7XXXXXXXXXX или 8XXXXXXXXXX)

    Returns:
        Информация о профиле клиента (имя, контакты, город, бизнес-данные, предпочтения)
        или "Профиль клиента не найден в базе данных."
    """
    try:
        return await get_client_profile_text(phone)
    except Exception as e:
        logger.error(f"Ошибка при получении профиля клиента: {e}", exc_info=True)
        return "Профиль клиента не найден в базе данных."


@tool
async def get_client_orders(phone: str) -> str:
    """Получает историю заказов клиента из базы данных.

    Возвращает список всех заказов клиента с информацией о товарах, ценах, весе,
    датах и пунктах назначения.

    Args:
        phone: Номер телефона клиента (в формате +7XXXXXXXXXX или 8XXXXXXXXXX)

    Returns:
        Строка с отформатированным списком заказов или "Заказы не найдены."
    """
    try:
        orders = await get_client_orders_from_db(phone)

        if not orders:
            return "Заказы не найдены."

        orders_list = []
        for order in orders:
            order_info = []
            if order.title:
                order_info.append(f"Товар: {order.title}")
            if order.created_at:
                order_info.append(f"Дата: {order.created_at}")
            if order.weight_kg is not None:
                order_info.append(f"Вес (кг): {order.weight_kg}")
            if order.price_out is not None:
                order_info.append(f"Цена: {order.price_out}")
            if order.destination:
                order_info.append(f"Пункт назначения: {order.destination}")
            
            if order_info:
                orders_list.append("\n".join(order_info))

        result_text = "\n\n---\n\n".join(orders_list)
        return f"Найдено заказов: {len(orders)}\n\n{result_text}"
    except Exception as e:
        logger.error(f"Ошибка при получении заказов: {e}", exc_info=True)
        return "Заказы не найдены."

