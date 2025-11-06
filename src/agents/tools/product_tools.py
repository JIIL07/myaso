"""Инструменты для работы с товарами."""

from __future__ import annotations

from typing import List
import logging
from langchain_core.tools import tool

from src.config.constants import VECTOR_SEARCH_LIMIT
from src.utils.retrievers import SupabaseVectorRetriever
from src.database.queries.products_queries import get_random_products as get_random_products_db

logger = logging.getLogger(__name__)


@tool
async def vector_search(query: str) -> str:
    """Ищет товары в базе данных по семантическому запросу пользователя.

    Использует векторный поиск (embeddings) для нахождения релевантных товаров.
    Возвращает отформатированный список товаров с их характеристиками.

    ИСПОЛЬЗУЙ ЭТОТ ИНСТРУМЕНТ ДЛЯ БОЛЬШИНСТВА ЗАПРОСОВ О ТОВАРАХ!

    Используй этот инструмент когда:
    - Пользователь запрашивает товары/ассортимент
    - Запрос содержит критерии поиска (тип мяса, часть туши, формат, вес, упаковка)
    - Пользователь спрашивает "Что у вас есть?", "Покажи мясо", "Какие стейки есть?"
    - Пользователь просит рекомендацию с критериями ("Что подходит для гриля?")
    - Пользователь спрашивает про товары от конкретного поставщика ("Что есть из продукции Мироторг", "товары от поставщика X", "что есть от Y")
    - Пользователь спрашивает про товары из конкретного региона ("мясо из региона Z", "товары из Сибири")
    - Запрос содержит название поставщика, региона или другие текстовые атрибуты товара
    - Запрос НЕ содержит числовых условий (цена, вес, скидка с числами)

    НЕ используй если:
    - Запрос содержит только подтверждение/отказ ("Да", "Нет", "Ок")
    - Обсуждаются сервисные темы (доставка, оплата, расписание)
    - Пользователь уточняет детали уже известного товара
    - Запрос содержит ЧИСЛОВЫЕ условия (цена меньше X, вес больше Y, скидка больше Z) - для этого используй generate_sql_from_text + execute_sql_request

    Args:
        query: Текстовый запрос пользователя о товарах/ассортименте

    Returns:
        Строка с отформатированным списком найденных товаров и их характеристиками
    """
    retriever = SupabaseVectorRetriever()

    try:
        documents = await retriever.get_relevant_documents(query, k=VECTOR_SEARCH_LIMIT)
    except Exception as e:
        logger.error(f"Ошибка при поиске по запросу '{query}': {e}", exc_info=True)
        return "Товары по вашему запросу не найдены."

    if not documents:
        return "Товары по вашему запросу не найдены."

    products_list = []
    for doc in documents:
        metadata = doc.metadata
        product_info = [
            f"Название: {metadata.get('title', 'Не указано')}",
            f"Поставщик: {metadata.get('supplier_name', 'Не указано')}",
            f"Регион: {metadata.get('from_region', 'Не указано')}",
            f"Цена за кг: {metadata.get('order_price_kg', 'Не указано')}",
            f"Минимальный заказ (кг): {metadata.get('min_order_weight_kg', 'Не указано')}",
            f"Охлаждённый/Замороженный: {metadata.get('cooled_or_frozen', 'Не указано')}",
            f"Полуфабрикат: {'Да' if metadata.get('ready_made') else 'Нет'}",
            f"Тип упаковки: {metadata.get('package_type', 'Не указано')}",
            f"Скидка: {metadata.get('discount', 'Не указано')}",
        ]
        products_list.append(
            "\n".join([info for info in product_info if "Не указано" not in info])
        )

    result = "\n\n---\n\n".join(products_list)
    return f"Найдено товаров: {len(documents)}\n\n{result}"


@tool
async def get_random_products(limit: int = 10) -> str:
    """Получает случайные товары из ассортимента.

    FALLBACK инструмент - используй когда другие поиски не дали результатов!

    Используй этот инструмент когда:
    - vector_search вернул "Товары по вашему запросу не найдены"
    - execute_sql_request вернул "Товары по указанным условиям не найдены"
    - Пользователь спрашивает "Что у вас есть?" и нужно показать любой ассортимент
    - Нужно показать примеры товаров из ассортимента
    - Все остальные инструменты поиска не дали результатов

    НЕ используй если:
    - vector_search или execute_sql_request уже нашли товары
    - Есть конкретный запрос, который можно обработать другими инструментами

    Это инструмент последней надежды - используй его только когда ничего не найдено!

    Args:
        limit: Количество товаров для возврата (по умолчанию 10, максимум 20)

    Returns:
        Строка с отформатированным списком случайных товаров
    """
    if limit > 20:
        limit = 20

    try:
        json_result = await get_random_products_db(limit)

        if not json_result:
            return "Товары не найдены."

        products_list = []
        for product in json_result:
            product_info = [
                f"Название: {product.get('title', 'Не указано')}",
                f"Поставщик: {product.get('supplier_name', 'Не указано')}",
                f"Регион: {product.get('from_region', 'Не указано')}",
                f"Цена за кг: {product.get('order_price_kg', 'Не указано')}",
                f"Минимальный заказ (кг): {product.get('min_order_weight_kg', 'Не указано')}",
            ]
            products_list.append(
                "\n".join([info for info in product_info if "Не указано" not in info])
            )

        result_text = "\n\n---\n\n".join(products_list)
        return f"Найдено товаров: {len(json_result)}\n\n{result_text}"

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных."
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}"

