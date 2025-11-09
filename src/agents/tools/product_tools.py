"""Инструменты для работы с товарами."""

from __future__ import annotations

from typing import List, Optional, Dict
import json
import logging
from langchain_core.tools import tool

from src.config.constants import VECTOR_SEARCH_LIMIT
from src.utils.retrievers import SupabaseVectorRetriever
from src.database.queries.products_queries import get_random_products as get_random_products_db
from src.utils.prompts import get_all_system_values

logger = logging.getLogger(__name__)


@tool
async def vector_search(query: str, require_photo: bool = False) -> str:
    """Семантический поиск товаров по текстовому запросу (векторный поиск).

    НАЗНАЧЕНИЕ: Семантический поиск товаров по текстовому запросу (векторный поиск)

    ИСПОЛЬЗУЙ ДЛЯ:
    - Текстовые запросы: "что у вас есть?", "покажи мясо", "какие стейки?"
    - Поиск по типу/части: "грудинка свиная", "говядина", "стейки", "полуфабрикаты"
    - Поиск по поставщику: "товары от Коралл", "продукция Мироторг", "весь коралл", "покажи весь коралл"
    - Поиск по региону: "мясо из Сибири", "товары из Бурятии"
    - Комбинации текстовых критериев: "свинина охлажденная", "стейки от Коралл"

    НЕ ИСПОЛЬЗУЙ ДЛЯ:
    - Числовые условия: "цена меньше 100" → используй generate_sql_from_text
    - Условия по весу: "вес больше 5 кг" → используй generate_sql_from_text
    - Условия по скидке: "скидка больше 15%" → используй generate_sql_from_text

    ПАРАМЕТР require_photo:
    - require_photo=True: Используй когда клиент запрашивает фото (например, "отправь фото грудинки")
      В этом случае будут возвращены ТОЛЬКО товары с фотографиями (photo IS NOT NULL AND photo != '')
      После поиска с require_photo=True, обязательно вызови show_product_photos для отправки фото
    - require_photo=False: По умолчанию, возвращаются все товары независимо от наличия фото
      Используй когда клиент просто спрашивает о товарах без запроса на фото

    ПРИМЕРЫ ПРАВИЛЬНОГО ИСПОЛЬЗОВАНИЯ:
    - Запрос: "отправь фото грудинки свиной" → vector_search(query="грудинка свиная", require_photo=True)
    - Запрос: "хочу увидеть фото стейков" → vector_search(query="стейки", require_photo=True)
    - Запрос: "покажи товары от Коралл" → vector_search(query="товары от Коралл", require_photo=False)

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
    for doc in documents:
        metadata = doc.metadata
        product_id = metadata.get('id')
        if product_id:
            product_ids.append(product_id)

        title = metadata.get('title', 'Не указано')
        supplier = metadata.get('supplier_name', '')
        price = metadata.get('order_price_kg', '')
        region = metadata.get('from_region', '')
        has_photo = bool(metadata.get('photo') and metadata.get('photo').strip())
        
        product_lines = [f"📦 {title}"]
        if supplier and supplier != 'Не указано':
            product_lines.append(f"   Поставщик: {supplier}")
        if price and price != 'Не указано':
            product_lines.append(f"   Цена: {price}₽/кг")
        if region and region != 'Не указано':
            product_lines.append(f"   Регион: {region}")
        if require_photo and has_photo:
            product_lines.append(f"   📷 Есть фото")
        
        products_list.append("\n".join(product_lines))

    result = "\n\n".join(products_list)
    more_text = "\n\n⚠️ В базе данных есть ещё товары, показываем первые 50. Используйте более конкретные критерии поиска для уточнения." if has_more else ""

    ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
    ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

    photo_note = " (только с фото)" if require_photo else ""
    return f"Найдено товаров: {len(documents)}{photo_note}{more_text}\n\n{result}{ids_section}"


@tool
async def get_random_products(limit: int = 10, require_photo: bool = False) -> str:
    """Получает случайные товары из ассортимента (FALLBACK инструмент).

    НАЗНАЧЕНИЕ: Получает случайные товары из ассортимента (FALLBACK инструмент)

    ИСПОЛЬЗУЙ ТОЛЬКО КОГДА:
    - vector_search вернул "Товары по вашему запросу не найдены"
    - execute_sql_request вернул "Товары по указанным условиям не найдены"
    - Все остальные инструменты поиска не дали результатов
    - Нужно показать примеры товаров из ассортимента когда ничего не найдено

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - vector_search или execute_sql_request уже нашли товары
    - Есть конкретный запрос, который можно обработать другими инструментами

    ВАЖНО: Это инструмент последней надежды! Всегда сначала пробуй vector_search или generate_sql_from_text + execute_sql_request.

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

        # Фильтруем товары с фото если требуется
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
        for product in json_result:
            product_id = product.get('id')
            if product_id:
                product_ids.append(product_id)

            product_info = [
                f"Название: {product.get('title', 'Не указано')}",
                f"Поставщик: {product.get('supplier_name', 'Не указано')}",
                f"Регион: {product.get('from_region', 'Не указано')}",
                f"Цена за кг: {product.get('order_price_kg', 'Не указано')}",
                f"Минимальный заказ (кг): {product.get('min_order_weight_kg', 'Не указано')}",
            ]
            if require_photo and product.get('photo'):
                product_info.append("📷 Есть фото")
            
            products_list.append(
                "\n".join([info for info in product_info if "Не указано" not in info])
            )

        result_text = "\n\n---\n\n".join(products_list)

        ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
        ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

        photo_note = " (только с фото)" if require_photo else ""
        return f"Найдено товаров: {len(json_result)}{photo_note}\n\n{result_text}{ids_section}"

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных."
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}"
