"""Tool: vector_search — semantic product search by natural language."""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.policy import get_agent_policy
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import (
    ERROR_MESSAGE_PRODUCTS_NOT_FOUND,
    ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND,
)
from src.toolkit import has_product_photo
from src.tools._contract import attach_product_ids, fail_response, ok_response
from src.tools._formatting import format_and_return_products, get_require_photo
from src.utils.retrievers import PostgresVectorRetriever

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def vector_search(
    query: str,
    k: int = 10,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, dict[str, object]]:
    """Семантический поиск товаров по текстовому описанию.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Клиент описывает товар словами
    - Поиск по общим характеристикам без точных числовых критериев

    НЕ ИСПОЛЬЗОВАТЬ:
    - Точное название товара -> get_product_by_title
    - Конкретные числовые критерии (цена, вес) -> execute_sql_query
    - Случайные товары без критериев -> get_random_products
    """
    policy = get_agent_policy()
    retriever = PostgresVectorRetriever()
    k = policy.clamp_vector_k(k)
    require_photo = get_require_photo(runtime)

    logger.debug(
        "[vector_search] query='%s', require_photo=%s, k=%s",
        query,
        require_photo,
        k,
    )

    try:
        search_k = policy.vector_photo_limit if require_photo else k
        documents = await retriever.get_relevant_documents(query, k=search_k)
    except Exception as e:
        logger.error("[vector_search] Ошибка поиска '%s': %s", query, e, exc_info=True)
        return fail_response(ERROR_MESSAGE_PRODUCTS_NOT_FOUND, error_code="search_failed")

    if not documents:
        return fail_response(ERROR_MESSAGE_PRODUCTS_NOT_FOUND, error_code="not_found")

    # --- Photo filter (at document level, before conversion) ---
    if require_photo:
        documents = [doc for doc in documents if has_product_photo(doc.metadata)]
        if not documents:
            logger.warning("[vector_search] Нет товаров с фото по запросу '%s'", query)
            return fail_response(
                ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND,
                error_code="not_found_with_photo",
            )

    documents = documents[:k]

    products = [
        {
            "id": doc.metadata.get("id"),
            "title": doc.metadata.get("title", "Не указано"),
            "supplier_name": doc.metadata.get("supplier_name"),
            "order_price_kg": doc.metadata.get("order_price_kg"),
            "from_region": doc.metadata.get("from_region"),
        }
        for doc in documents
    ]

    text, product_ids = await format_and_return_products(products, require_photo=False)
    artifact = attach_product_ids({"query": query, "k": k}, product_ids)
    return ok_response(text, artifact=artifact)
