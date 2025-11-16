"""Инструменты для работы с товарами."""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from src.config.constants import VECTOR_SEARCH_LIMIT
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

    ПАРАМЕТР require_photo:
    - require_photo=True: Используй когда клиент запрашивает фото (например, "отправь фото грудинки")
      В этом случае будут возвращены ТОЛЬКО товары с фотографиями (photo IS NOT NULL AND photo != '')
      После поиска с require_photo=True, обязательно вызови show_product_photos для отправки фото
    - require_photo=False: По умолчанию, возвращаются все товары независимо от наличия фото
      Используй когда клиент просто спрашивает о товарах без запроса на фото

    Args:
        query: Текстовый запрос пользователя о товарах
        require_photo: Если True, возвращает только товары с фотографиями (по умолчанию False)

    Returns:
        Список найденных товаров (до 50) с ID в секции [PRODUCT_IDS]
    """
    retriever = SupabaseVectorRetriever()

    try:
        k = (VECTOR_SEARCH_LIMIT + 1) * 3 if require_photo else VECTOR_SEARCH_LIMIT + 1
        documents = await retriever.get_relevant_documents(query, k=k)
    except Exception as e:
        logger.error(f"Ошибка при поиске по запросу '{query}': {e}", exc_info=True)
        return "Товары по вашему запросу не найдены."

    if not documents:
        return "Товары по вашему запросу не найдены."

    if require_photo:
        documents = [
            doc for doc in documents 
            if doc.metadata.get('photo') and doc.metadata.get('photo').strip()
        ]
        if not documents:
            return "Товары с фотографиями по вашему запросу не найдены."

    has_more = len(documents) > VECTOR_SEARCH_LIMIT
    documents = documents[:VECTOR_SEARCH_LIMIT]

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
        has_photo = bool(metadata.get('photo') and metadata.get('photo').strip())
        
        final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)
        
        product_lines = [f"📦 {title}"]
        product_lines.append(f"   Поставщик: {supplier}")
        if final_price != "Цена по запросу":
            product_lines.append(f"   Цена: {final_price}₽/кг")
        else:
            product_lines.append(f"   Цена: {final_price}")
        product_lines.append(f"   Регион: {region}")
        if require_photo and has_photo:
            product_lines.append(f"   📷 Есть фото")
        
        products_list.append("\n".join(product_lines))

    result_text = "\n\n".join(products_list)
    more_text = "\n\n⚠️ В базе данных есть ещё товары, показываем первые 50. Используйте более конкретные критерии поиска для уточнения." if has_more else ""

    ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
    ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

    photo_note = " (только с фото)" if require_photo else ""
    return f"Найдено товаров: {len(documents)}{photo_note}{more_text}\n\n{result_text}{ids_section}"


@tool
async def get_random_products(limit: int = 10, require_photo: bool = False) -> str:
    """Получает случайные товары из ассортимента (FALLBACK инструмент).

    НАЗНАЧЕНИЕ: Получает случайные товары из ассортимента (FALLBACK инструмент)

    ИСПОЛЬЗУЙ ТОЛЬКО КОГДА:
    - vector_search вернул "Товары по вашему запросу не найдены"
    - execute_sql_request вернул "Товары по указанным условиям не найдены"
    - Все остальные инструменты поиска не дали результатов
    - Нужно показать примеры товаров из ассортимента когда ничего не найдено

    ПАРАМЕТР require_photo:
    - require_photo=True: Используй когда клиент запрашивает фото
      В этом случае будут возвращены ТОЛЬКО товары с фотографиями
      После поиска с require_photo=True, обязательно вызови show_product_photos для отправки фото
    - require_photo=False: По умолчанию, возвращаются все товары независимо от наличия фото

    Args:
        limit: Количество товаров для возврата (по умолчанию 10, максимум 20)
        require_photo: Если True, возвращает только товары с фотографиями (по умолчанию False)

    Returns:
        Список случайных товаров (до 20) с ID в секции [PRODUCT_IDS]
    """
    if limit > 20:
        limit = 20

    try:
        json_result = await get_random_products_db(limit * 3 if require_photo else limit)

        if not json_result:
            return "Товары не найдены."

        if require_photo:
            json_result = [
                product for product in json_result
                if product.get('photo') and product.get('photo').strip()
            ]
            if not json_result:
                return "Товары с фотографиями не найдены."
            json_result = json_result[:limit]

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
            has_photo = bool(product.get('photo') and product.get('photo').strip())
            
            final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)
            
            product_lines = [f"📦 {title}"]
            product_lines.append(f"   Поставщик: {supplier}")
            if final_price != "Цена по запросу":
                product_lines.append(f"   Цена: {final_price}₽/кг")
            else:
                product_lines.append(f"   Цена: {final_price}")
            product_lines.append(f"   Регион: {region}")
            if require_photo and has_photo:
                product_lines.append(f"   📷 Есть фото")
            
            products_list.append("\n".join(product_lines))

        result_text = "\n\n".join(products_list)
        more_text = ""  # Для случайных товаров more_text не применим

        ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
        ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

        photo_note = " (только с фото)" if require_photo else ""
        return f"Найдено товаров: {len(json_result)}{photo_note}{more_text}\n\n{result_text}{ids_section}"

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных."
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}"
