"""Tool: get_product_by_title — find product by exact name."""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.queries.products_queries import get_product_by_title as get_product_by_title_db
from src.tools._formatting import format_and_return_products, get_require_photo

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def get_product_by_title(
    title: str,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, list[int]]:
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
            return "Товар '%s' не найден в базе данных." % title, []

        text, ids = await format_and_return_products(
            [product.model_dump()],
            require_photo=require_photo,
            no_photo_message="Товар '%s' найден, но не имеет фотографии." % title,
        )

        if ids:
            text = "Найден товар:\n\n" + text.split("\n\n", 1)[-1]

        return text, ids
    except Exception as e:
        logger.error(
            "[get_product_by_title] Ошибка поиска '%s': %s",
            title,
            e,
            exc_info=True,
        )
        return "Ошибка при поиске товара: %s" % e, []
