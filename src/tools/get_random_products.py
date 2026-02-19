"""Tool: get_random_products — show random products from the catalog."""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import ERROR_MESSAGE_DATABASE_NOT_CONFIGURED, PHOTO_SEARCH_LIMIT_MULTIPLIER
from src.queries.products_queries import get_random_products as get_random_products_db
from src.tools._formatting import (
    calculate_search_limit,
    format_and_return_products,
    get_require_photo,
)

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def get_random_products(
    limit: int = 10,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, list[int]]:
    """Случайные товары из каталога для демонстрации ассортимента.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - "покажи товары", "что у вас есть", "покажи примеры"
    - Клиент хочет увидеть ассортимент без конкретных требований

    НЕ ИСПОЛЬЗОВАТЬ:
    - Есть конкретные критерии поиска -> vector_search или execute_sql_query
    - Точное название товара -> get_product_by_title
    """
    require_photo = get_require_photo(runtime)

    try:
        search_limit = calculate_search_limit(limit, require_photo, PHOTO_SEARCH_LIMIT_MULTIPLIER)
        products = await get_random_products_db(search_limit)

        return await format_and_return_products(
            [p.model_dump() for p in products] if products else [],
            require_photo=require_photo,
            limit=limit,
        )
    except RuntimeError as e:
        logger.error("[get_random_products] Ошибка подключения к БД: %s", e)
        return ERROR_MESSAGE_DATABASE_NOT_CONFIGURED, []
    except Exception as e:
        logger.error("[get_random_products] Ошибка: %s", e, exc_info=True)
        return "Ошибка при получении товаров: %s" % e, []
