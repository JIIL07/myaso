"""Tool: get_product_by_title — find product by exact name."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.queries.products_queries import get_product_by_title as get_product_by_title_db
from src.tools.common._contract import attach_product_ids, fail_response, ok_response
from src.tools.common._formatting import format_and_return_products, get_require_photo

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def get_product_by_title(
    title: str,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, dict[str, Any]]:
    """Поиск товара по точному названию (регистронезависимо).

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Клиент называет точное название
    - Известно точное название из контекста разговора

    НЕ ИСПОЛЬЗОВАТЬ:
    - Описание общими словами -> vector_search
    - Поиск по нескольким критериям -> execute_sql_query
    """
    require_photo = get_require_photo(runtime)

    try:
        product = await get_product_by_title_db(title)
        if not product:
            return fail_response(
                "Товар '%s' не найден в базе данных." % title,
                error_code="not_found",
            )

        text, product_ids = await format_and_return_products(
            [product.model_dump()],
            require_photo=require_photo,
            no_photo_message="Товар '%s' найден, но не имеет фотографии." % title,
        )

        if product_ids:
            text = "Найден товар:\n\n" + text.split("\n\n", 1)[-1]

        return ok_response(
            text,
            artifact=attach_product_ids({"title": title}, product_ids),
        )
    except Exception as e:
        logger.error(
            "[get_product_by_title] Ошибка поиска '%s': %s",
            title,
            e,
            exc_info=True,
        )
        return fail_response("Ошибка при поиске товара: %s" % e, error_code="unexpected_error")

