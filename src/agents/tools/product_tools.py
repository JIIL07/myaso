"""Инструменты для работы с товарами."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from src.database.queries.products_queries import (
    get_random_products as get_random_products_db,
)
from src.utils.field_normalizer import normalize_field_value
from src.utils.price_calculator import calculate_final_price
from src.utils.prompts import get_all_system_values
from src.utils.retrievers import SupabaseVectorRetriever

logger = logging.getLogger(__name__)


@tool
async def vector_search(query: str, require_photo: bool = False) -> str:
    """Семантический поиск товаров по текстовому запросу (векторный поиск).

    НАЗНАЧЕНИЕ: Семантический поиск товаров по текстовому запросу (векторный поиск)

    ИСПОЛЬЗУЙ ДЛЯ:
    - Текстовые запросы по названию/типу
    - Поиск по текстовым атрибутам (без чисел)
    - Семантический поиск (синонимы, контекст)

    Args:
        query: Текстовый запрос пользователя о товарах
        require_photo: Если True, возвращает только товары с фотографиями

    Returns:
        Список найденных товаров с ID в секции [PRODUCT_IDS]
    """
    retriever = SupabaseVectorRetriever()

    try:
        # Получаем до 250 товаров для фильтрации по фото
        # Сортировка по релевантности сохраняется в SQL запросе
        documents = await retriever.get_relevant_documents(query, k=250)
    except Exception as e:
        logger.error(f"Ошибка при поиске по запросу '{query}': {e}", exc_info=True)
        return "Товары по вашему запросу не найдены."

    if not documents:
        return "Товары по вашему запросу не найдены."

    # Фильтрация по наличию фото ДО обработки результатов (если требуется)
    # Сортировка по релевантности (distance) сохраняется, так как документы уже отсортированы
    if require_photo:
        documents = [
            doc for doc in documents
            if doc.metadata.get('photo')
        ]
        if not documents:
            return "Товары с фотографиями по вашему запросу не найдены."

    # Лимиты управляются через SYSTEM_PROMPT из БД, не ограничиваем здесь
    # Агент сам выберет самые релевантные товары из отсортированного списка
    products_list = []
    product_ids = []
    
    system_vars = await get_all_system_values()
    
    for doc in documents:
        metadata = doc.metadata
        product_id = metadata.get('id')
        if product_id:
            product_ids.append(product_id)

        title = metadata.get('title', 'Не указано')
        supplier = normalize_field_value(metadata.get('supplier_name'), 'text')
        order_price = metadata.get('order_price_kg')
        region = normalize_field_value(metadata.get('from_region'), 'text')
        
        final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)
        
        product_lines = [f"📦 {title}"]
        product_lines.append(f"   Поставщик: {supplier}")
        if final_price != "Цена по запросу":
            product_lines.append(f"   Цена: {final_price}₽/кг")
        else:
            product_lines.append(f"   Цена: {final_price}")
        product_lines.append(f"   Регион: {region}")
        
        products_list.append("\n".join(product_lines))

    result_text = "\n\n".join(products_list)
    more_text = ""  # Для vector_search more_text не применим

    ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
    ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

    return f"Найдено товаров: {len(documents)}{more_text}\n\n{result_text}{ids_section}"


@tool
async def get_random_products(limit: int = 10) -> str:
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
        Список случайных товаров с ID в секции [PRODUCT_IDS]
    """
    # Лимиты управляются через SYSTEM_PROMPT из БД, принимаем любой limit

    try:
        json_result = await get_random_products_db(limit)

        if not json_result:
            return "Товары не найдены."

        products_list = []
        product_ids = []
        
        system_vars = await get_all_system_values()
        
        for product in json_result:
            product_id = product.get('id')
            if product_id:
                product_ids.append(product_id)

            title = product.get('title', 'Не указано')
            supplier = normalize_field_value(product.get('supplier_name'), 'text')
            order_price = product.get('order_price_kg')
            region = normalize_field_value(product.get('from_region'), 'text')
            
            final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)
            
            product_lines = [f"📦 {title}"]
            product_lines.append(f"   Поставщик: {supplier}")
            if final_price != "Цена по запросу":
                product_lines.append(f"   Цена: {final_price}₽/кг")
            else:
                product_lines.append(f"   Цена: {final_price}")
            product_lines.append(f"   Регион: {region}")
            
            products_list.append("\n".join(product_lines))

        result_text = "\n\n".join(products_list)
        more_text = ""  # Для случайных товаров more_text не применим

        ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
        ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

        return f"Найдено товаров: {len(json_result)}{more_text}\n\n{result_text}{ids_section}"

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных."
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}"
