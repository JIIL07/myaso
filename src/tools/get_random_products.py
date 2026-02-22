"""Tool: get_random_products — show random products from the catalog."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.policy import get_agent_policy
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import ERROR_MESSAGE_DATABASE_NOT_CONFIGURED
from src.queries.products_queries import get_random_products as get_random_products_db
from src.tools._contract import attach_product_ids, fail_response, ok_response
from src.tools._formatting import calculate_search_limit, format_and_return_products, get_require_photo

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def get_random_products(
    limit: int = 10,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, dict[str, Any]]:
    """Случайные товары из каталога для демонстрации ассортимента.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - "покажи товары", "что у вас есть", "покажи примеры"
    - Клиент хочет увидеть ассортимент без конкретных требований

    НЕ ИСПОЛЬЗОВАТЬ:
    - Есть конкретные критерии поиска -> vector_search или execute_sql_query
    - Точное название товара -> get_product_by_title
    """
    policy = get_agent_policy()
    limit = policy.clamp_sql_limit(limit)
    require_photo = get_require_photo(runtime)

    try:
        search_limit = calculate_search_limit(limit, require_photo)
        products = await get_random_products_db(search_limit)

        text, product_ids = await format_and_return_products(
            [p.model_dump() for p in products] if products else [],
            require_photo=require_photo,
            limit=limit,
        )
        return ok_response(
            text,
            artifact=attach_product_ids(
                {"limit": limit, "require_photo": require_photo},
                product_ids,
            ),
        )
    except RuntimeError as e:
        logger.error("[get_random_products] Ошибка подключения к БД: %s", e)
        return fail_response(
            ERROR_MESSAGE_DATABASE_NOT_CONFIGURED,
            error_code="database_not_configured",
        )
    except Exception as e:
        logger.error("[get_random_products] Ошибка: %s", e, exc_info=True)
        return fail_response("Ошибка при получении товаров: %s" % e, error_code="unexpected_error")
