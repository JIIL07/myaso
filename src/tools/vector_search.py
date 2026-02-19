"""Tool: vector_search — semantic product search by natural language."""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import (
    ERROR_MESSAGE_PRODUCTS_NOT_FOUND,
    ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND,
    MAX_VECTOR_SEARCH_RESULTS,
    VECTOR_SEARCH_PHOTO_LIMIT,
)
from src.tools._formatting import format_and_return_products, get_require_photo
from src.utils.formatters.formatters import has_photo
from src.utils.retrievers import SupabaseVectorRetriever

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def vector_search(
    query: str,
    k: int = 10,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, list[int]]:
    """Семантический поиск товаров по текстовому описанию.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Клиент описывает товар словами
    - Поиск по общим характеристикам без точных числовых критериев

    НЕ ИСПОЛЬЗОВАТЬ:
    - Точное название товара -> get_product_by_title
    - Конкретные числовые критерии (цена, вес) -> execute_sql_query
    - Случайные товары без критериев -> get_random_products
    """
    retriever = SupabaseVectorRetriever()
    k = min(k, MAX_VECTOR_SEARCH_RESULTS)
    require_photo = get_require_photo(runtime)

    logger.debug(
        "[vector_search] query='%s', require_photo=%s, k=%s",
        query,
        require_photo,
        k,
    )

    try:
        search_k = VECTOR_SEARCH_PHOTO_LIMIT if require_photo else k
        documents = await retriever.get_relevant_documents(query, k=search_k)
    except Exception as e:
        logger.error("[vector_search] Ошибка поиска '%s': %s", query, e, exc_info=True)
        return ERROR_MESSAGE_PRODUCTS_NOT_FOUND, []

    if not documents:
        return ERROR_MESSAGE_PRODUCTS_NOT_FOUND, []

    # --- Photo filter (at document level, before conversion) ---
    if require_photo:
        documents = [doc for doc in documents if has_photo(doc.metadata)]
        if not documents:
            logger.warning("[vector_search] Нет товаров с фото по запросу '%s'", query)
            return ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND, []

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

    # Photo filtering already done above, pass require_photo=False to avoid double-filtering
    return await format_and_return_products(products, require_photo=False)
