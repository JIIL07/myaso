"""Инструменты для работы с клиентами."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.database.queries.clients_queries import get_client_profile_text
from src.database.queries.orders_queries import (
    get_client_orders as get_client_orders_from_db,
    get_last_order as get_last_order_from_db,
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

    НАЗНАЧЕНИЕ: Получает историю заказов клиента из базы данных

    ИСПОЛЬЗУЙ КОГДА:
    - Клиент спрашивает про свои заказы
    - Клиент хочет узнать историю покупок
    - Нужна информация о том, что клиент заказывал ранее
    - Для персонализации рекомендаций на основе истории заказов

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Клиент спрашивает про товары, а не про заказы
    - Информация о заказах не нужна для ответа

    Args:
        phone: Номер телефона клиента (в формате +7XXXXXXXXXX или 8XXXXXXXXXX)

    Returns:
        Строка с отформатированным списком заказов с информацией о товарах, ценах, весе,
        датах и пунктах назначения, или "Заказы не найдены."
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


@tool
async def get_last_order(phone: str) -> str:
    """Получает информацию о последнем заказе клиента.

    НАЗНАЧЕНИЕ: Получает информацию о последнем заказе клиента

    ИСПОЛЬЗУЙ КОГДА:
    - Клиент спрашивает про последний заказ
    - Нужно узнать что клиент заказывал недавно
    - Для персонализации рекомендаций на основе последнего заказа
    - Когда клиент спрашивает "что я заказывал в последний раз?"

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Клиент спрашивает про всю историю заказов (используй get_client_orders)
    - Информация о заказах не нужна для ответа

    Args:
        phone: Номер телефона клиента (в формате +7XXXXXXXXXX или 8XXXXXXXXXX)

    Returns:
        Информация о последнем заказе или "Заказы не найдены."
    """
    try:
        order = await get_last_order_from_db(phone)
        if not order:
            return "Заказы не найдены."

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

        return "\n".join(order_info) if order_info else "Информация о заказе отсутствует."
    except Exception as e:
        logger.error(f"Ошибка при получении последнего заказа: {e}", exc_info=True)
        return "Заказы не найдены."

