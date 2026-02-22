"""Tool: get_client_orders — retrieve client order history."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.queries.orders_queries import get_client_orders as get_client_orders_from_db
from src.toolkit import has_client_phone
from src.tools._contract import fail_response, ok_response

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def get_client_orders(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> tuple[str, dict[str, Any]]:
    """История заказов клиента: товары, даты, вес, цены, пункты назначения.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - "что я заказывал", "мои заказы", "история покупок"
    - Нужен анализ прошлых заказов для рекомендаций

    НЕ ИСПОЛЬЗОВАТЬ:
    - Вопрос о товарах каталога -> инструменты поиска
    - Нужна общая информация о клиенте -> get_client_profile
    """
    try:
        client_phone = runtime.context.client_phone

        if not has_client_phone(client_phone):
            return fail_response("Номер телефона клиента не указан.", error_code="missing_phone")

        orders = await get_client_orders_from_db(client_phone)

        if not orders:
            return fail_response(
                "Заказы не найдены.",
                error_code="not_found",
                artifact={"order_ids": [], "count": 0},
            )

        orders_list = []
        order_ids = []
        for order in orders:
            order_info = []
            if hasattr(order, "id") and order.id:
                order_ids.append(order.id)
            if order.title:
                order_info.append("Товар: %s" % order.title)
            if order.created_at:
                order_info.append("Дата: %s" % order.created_at)
            if order.weight_kg is not None:
                order_info.append("Вес (кг): %s" % order.weight_kg)
            if order.price_out is not None:
                order_info.append("Цена: %s" % order.price_out)
            if order.destination:
                order_info.append("Пункт назначения: %s" % order.destination)

            if order_info:
                orders_list.append("\n".join(order_info))

        result_text = "\n\n---\n\n".join(orders_list)
        artifact = {"order_ids": order_ids, "count": len(orders)}
        return ok_response("Найдено заказов: %d\n\n%s" % (len(orders), result_text), artifact=artifact)
    except Exception as e:
        logger.error("[get_client_orders] Ошибка: %s", e, exc_info=True)
        return fail_response(
            "Произошла ошибка при получении заказов. "
            "Попробуйте позже или обратитесь в поддержку.",
            error_code="orders_error",
        )
