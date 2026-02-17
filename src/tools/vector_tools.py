"""Инструменты для векторного поиска товаров."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.utils.retrievers import SupabaseVectorRetriever
from src.services.ai.constants import DEFAULT_FIELD_VALUE
from src.services.ai.prompt import get_all_system_values
from src.utils.formatters.formatters import format_products_list
from src.tools.utils import get_require_photo_from_runtime
from src.services.database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

ERROR_MESSAGE_PRODUCTS_NOT_FOUND = "Товары по вашему запросу не найдены."
ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND = "Товары с фотографиями по вашему запросу не найдены."

VECTOR_SEARCH_PHOTO_LIMIT = 250
MAX_VECTOR_SEARCH_RESULTS = 50


@tool(response_format="content_and_artifact")
async def vector_search(
    query: str,
    k: int = 10,
    require_photo: bool = False,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> Tuple[str, List[int]]:
    """Семантический поиск товаров по текстовому запросу.

    Использует векторный поиск для нахождения товаров по семантическому сходству.
    Требование наличия фотографий автоматически берется из state агента.
    Если установлено требование фото (через set_photo_requirement),
    будут возвращены только товары с фотографиями.

    Args:
        query: Текстовый запрос пользователя о товарах
        k: Количество результатов для возврата (по умолчанию 10, максимум 50)
        require_photo: Игнорируется, значение берется из state агента
        runtime: ToolRuntime для доступа к context и state

    Returns:
        Кортеж из двух элементов:
        - Строка с результатами поиска
        - Список ID найденных товаров
    """
    from src.utils.formatters.formatters import has_photo

    retriever = SupabaseVectorRetriever()
    k = min(k, MAX_VECTOR_SEARCH_RESULTS)
    require_photo = get_require_photo_from_runtime(runtime)

    logger.debug(
        "[vector_search] Поиск по запросу '%s', require_photo=%s, k=%s",
        query,
        require_photo,
        k,
    )

    try:
        search_k = VECTOR_SEARCH_PHOTO_LIMIT if require_photo else k
        documents = await retriever.get_relevant_documents(query, k=search_k)
    except Exception as e:
        logger.error(
            "[vector_search] Ошибка при поиске по запросу '%s': %s",
            query,
            e,
            exc_info=True,
        )
        return ERROR_MESSAGE_PRODUCTS_NOT_FOUND, []

    if not documents:
        return ERROR_MESSAGE_PRODUCTS_NOT_FOUND, []

    if require_photo:
        documents = [
            doc for doc in documents if has_photo(doc.metadata)
        ]
        if not documents:
            logger.warning(
                "[vector_search] Не найдено товаров с фото по запросу '%s'", query
            )
            return ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND, []

    documents = documents[:k]
    products = []
    for doc in documents:
        metadata = doc.metadata
        products.append(
            {
                "id": metadata.get("id"),
                "title": metadata.get("title", "Не указано"),
                "supplier_name": metadata.get("supplier_name"),
                "order_price_kg": metadata.get("order_price_kg"),
                "from_region": metadata.get("from_region"),
            }
        )

    system_vars = await get_all_system_values()
    result_text, product_ids = await format_products_list(products, system_vars)
    return f"Найдено товаров: {len(products)}\n\n{result_text}", product_ids
