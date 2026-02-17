"""Инструменты для работы с клиентами."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.queries.clients_queries import get_client_profile_text
from src.queries.orders_queries import (
    get_client_orders as get_client_orders_from_db,
)
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.utils.validators import validate_client_phone

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def get_client_profile(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> Tuple[str, Dict]:
    """Возвращает текстовый профиль клиента и метаданные по номеру телефона из context."""
    try:
        client_phone = runtime.context.client_phone
        
        if not validate_client_phone(client_phone):
            error_text = "Номер телефона клиента не указан."
            return error_text, {}
        
        profile_text = await get_client_profile_text(client_phone)
        artifact = {
            "phone": client_phone,
            "profile_retrieved": bool(profile_text and profile_text.strip()),
        }
        return profile_text, artifact
    except Exception as e:
        error_text = (
            "Произошла ошибка при получении профиля клиента. "
            "Попробуйте позже или обратитесь в поддержку."
        )
        return error_text, {}


@tool(response_format="content_and_artifact")
async def get_client_orders(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> Tuple[str, Dict]:
    """Возвращает историю заказов клиента и артефакт с идентификаторами заказов."""
    try:
        client_phone = runtime.context.client_phone
        
        if not validate_client_phone(client_phone):
            error_text = "Номер телефона клиента не указан."
            return error_text, {}
        
        orders = await get_client_orders_from_db(client_phone)

        if not orders:
            return "Заказы не найдены.", {"order_ids": [], "count": 0}

        orders_list = []
        order_ids = []
        for order in orders:
            order_info = []
            if hasattr(order, 'id') and order.id:
                order_ids.append(order.id)
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
        artifact = {"order_ids": order_ids, "count": len(orders)}
        return f"Найдено заказов: {len(orders)}\n\n{result_text}", artifact
    except Exception as e:
        error_text = (
            "Произошла ошибка при получении заказов. "
            "Попробуйте позже или обратитесь в поддержку."
        )
        return error_text, {}
