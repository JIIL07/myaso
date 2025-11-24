"""Инструменты для работы с товарами."""

from __future__ import annotations

import logging
from typing import List, Tuple

from langchain_core.tools import tool

from src.database.queries.products_queries import (
    get_random_products as get_random_products_db,
)
from src.utils.prompts import get_all_system_values
from src.utils.retrievers import SupabaseVectorRetriever
from src.utils.product_formatter import format_products_list
from src.agents.tools.context_tools import get_require_photo
from src.agents.tools.context_vars import get_client_phone

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def vector_search(query: str, k: int = 10, require_photo: bool = False) -> Tuple[str, List[int]]:
    """Семантический поиск товаров по текстовому запросу (векторный поиск).

    НАЗНАЧЕНИЕ: Семантический поиск товаров по текстовому запросу (векторный поиск)

    ИСПОЛЬЗУЙ ДЛЯ:
    - Текстовые запросы по названию/типу товара
    - Поиск по текстовым атрибутам (без чисел)
    - Семантический поиск (синонимы, контекст)

    Args:
        query: Текстовый запрос пользователя о товарах
        k: Количество результатов для возврата (по умолчанию 10, максимум 50)
        require_photo: Если True, возвращает только товары с фотографиями.
                      Если False, используется значение из контекста агента (если установлено)

    Returns:
        Кортеж (текст с результатами, список ID товаров как artifact)
    """
    # Используем контекст если require_photo не указан явно
    if not require_photo:
        # Пытаемся получить из контекста (требует client_phone, но мы не знаем его здесь)
        pass
    retriever = SupabaseVectorRetriever()

    # Ограничиваем k максимумом 50
    k = min(k, 50)

    try:
        search_k = 250 if require_photo else k
        documents = await retriever.get_relevant_documents(query, k=search_k)
        # Получаем больше товаров для фильтрации по фото если требуется
    except Exception as e:
        logger.error(f"Ошибка при поиске по запросу '{query}': {e}", exc_info=True)
        return "Товары по вашему запросу не найдены.", []

    if not documents:
        return "Товары по вашему запросу не найдены.", []

    # Фильтрация по наличию фото ДО обработки результатов (если требуется)
    if require_photo:
        documents = [
            doc for doc in documents
            if doc.metadata.get('photo')
        ]
        if not documents:
            return "Товары с фотографиями по вашему запросу не найдены.", []

    documents = documents[:k]
    products = []
    for doc in documents:
        metadata = doc.metadata
        products.append({
            "id": metadata.get('id'),
            "title": metadata.get('title', 'Не указано'),
            "supplier_name": metadata.get('supplier_name'),
            "order_price_kg": metadata.get('order_price_kg'),
            "from_region": metadata.get('from_region'),
        })

    result_text, product_ids = await format_products_list(products)
    
    # Возвращаем кортеж: (текст, artifact с product_ids)
    return f"Найдено товаров: {len(documents)}\n\n{result_text}", product_ids


@tool(response_format="content_and_artifact")
async def get_random_products(limit: int = 10) -> Tuple[str, List[int]]:
    """Получает случайные товары из ассортимента (FALLBACK инструмент).

    НАЗНАЧЕНИЕ: Получает случайные товары из ассортимента (FALLBACK инструмент)

    ИСПОЛЬЗУЙ ТОЛЬКО КОГДА:
    - vector_search вернул "Товары по вашему запросу не найдены"
    - execute_sql_query вернул "Товары по указанным условиям не найдены" или "По указанному запросу ничего не найдено"
    - Все остальные инструменты поиска не дали результатов
    - Нужно показать примеры товаров из ассортимента когда ничего не найдено

    Args:
        limit: Количество товаров для возврата (по умолчанию 10)

    Returns:
        Кортеж (текст с результатами, список ID товаров как artifact)
    """
    # Лимиты управляются через SYSTEM_PROMPT из БД, принимаем любой limit

    try:
        json_result = await get_random_products_db(limit)

        if not json_result:
            return "Товары не найдены.", []

        result_text, product_ids = await format_products_list(json_result)

        # Возвращаем кортеж: (текст, artifact с product_ids)
        return f"Найдено товаров: {len(json_result)}\n\n{result_text}", product_ids

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных.", []
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}", []
