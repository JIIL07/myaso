"""Инструменты для работы с товарами."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from langchain_core.tools import tool

from src.config.messages_constants import (
    ERROR_MESSAGE_DATABASE_NOT_CONFIGURED,
    ERROR_MESSAGE_PRODUCTS_NOT_FOUND,
    ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND,
)
from src.database.queries.products_queries import (
    get_random_products as get_random_products_db,
    get_product_by_title as get_product_by_title_db,
)
from src.database.queries.orders_queries import (
    get_client_orders as get_client_orders_db,
)
from src.utils.prompts import get_all_system_values
from src.utils.retrievers import SupabaseVectorRetriever
from src.utils.product_formatter import format_products_list
from src.utils.price_calculator import calculate_final_price
from src.utils import records_to_json
from src.database import get_pool
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
    if not require_photo:
        # Пытаемся получить из контекста (требует client_phone, но мы не знаем его здесь)
        pass
    retriever = SupabaseVectorRetriever()

    k = min(k, 50)

    try:
        search_k = 250 if require_photo else k
        documents = await retriever.get_relevant_documents(query, k=search_k)
        # Получаем больше товаров для фильтрации по фото если требуется
    except Exception as e:
        logger.error(f"Ошибка при поиске по запросу '{query}': {e}", exc_info=True)
        return ERROR_MESSAGE_PRODUCTS_NOT_FOUND, []

    if not documents:
        return ERROR_MESSAGE_PRODUCTS_NOT_FOUND, []

    # Фильтрация по наличию фото ДО обработки результатов (если требуется)
    if require_photo:
        documents = [
            doc for doc in documents
            if doc.metadata.get('photo')
        ]
        if not documents:
            return ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND, []

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
    return f"Найдено товаров: {len(products)}\n\n{result_text}", product_ids


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

    try:
        products = await get_random_products_db(limit)

        if not products:
            return "Товары не найдены.", []

        products_dict = [product.model_dump() for product in products]
        result_text, product_ids = await format_products_list(products_dict)

        return f"Найдено товаров: {len(products_dict)}\n\n{result_text}", product_ids

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return ERROR_MESSAGE_DATABASE_NOT_CONFIGURED, []
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}", []


@tool(response_format="content_and_artifact")
async def get_product_by_title(title: str) -> Tuple[str, List[int]]:
    """Находит товар по точному названию.

    НАЗНАЧЕНИЕ: Поиск товара по точному названию (без семантического поиска)

    ИСПОЛЬЗУЙ ДЛЯ:
    - Когда клиент знает точное название товара
    - Для быстрого поиска конкретного товара
    - Когда нужен один конкретный товар, а не список похожих

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Клиент ищет товары по описанию или типу (используй vector_search)
    - Нужно найти несколько похожих товаров (используй vector_search)
    - Название товара неточное или частичное (используй vector_search)

    Args:
        title: Точное название товара

    Returns:
        Кортеж (текст с результатами, список ID товаров как artifact)
    """
    try:
        product = await get_product_by_title_db(title)
        if not product:
            return f"Товар '{title}' не найден в базе данных.", []

        products_dict = [product.model_dump()]
        result_text, product_ids = await format_products_list(products_dict)

        return f"Найден товар:\n\n{result_text}", product_ids
    except Exception as e:
        logger.error(f"Ошибка при поиске товара по названию '{title}': {e}", exc_info=True)
        return f"Ошибка при поиске товара: {str(e)}", []


@tool(response_format="content_and_artifact")
async def find_similar_products(product_id: int, k: int = 10) -> Tuple[str, List[int]]:
    """Находит товары, похожие на указанный товар.

    НАЗНАЧЕНИЕ: Поиск товаров, похожих на указанный товар

    ИСПОЛЬЗУЙ ДЛЯ:
    - Когда клиент просит "что-то похожее на товар X"
    - Для поиска альтернатив конкретному товару
    - Когда нужен товар с похожими характеристиками
    - Когда клиент хочет найти замену товару

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Клиент ищет товары по текстовому запросу (используй vector_search)
    - Нужен точный товар по названию (используй get_product_by_title)

    Args:
        product_id: ID товара для поиска похожих
        k: Количество похожих товаров (по умолчанию 10, максимум 20)

    Returns:
        Кортеж (текст с результатами, список ID товаров как artifact)
    """
    k = min(k, 20)

    try:
        from src.utils import get_supabase_client
        supabase = await get_supabase_client()

        # Получаем товар
        result = (
            await supabase.table("products")
            .select("*")
            .eq("id", product_id)
            .execute()
        )

        if not result.data:
            return f"Товар с ID {product_id} не найден.", []

        product = result.data[0]
        product_title = product.get("title", "")

        if not product_title:
            return f"У товара с ID {product_id} нет названия для поиска похожих товаров.", []

        # Используем векторный поиск для поиска похожих товаров
        retriever = SupabaseVectorRetriever()
        documents = await retriever.get_relevant_documents(product_title, k=k + 1)

        # Исключаем сам товар из результатов
        similar_products = [
            doc.metadata for doc in documents
            if doc.metadata.get('id') != product_id
        ][:k]

        if not similar_products:
            return f"Не найдено похожих товаров для '{product_title}'.", []

        result_text, product_ids = await format_products_list(similar_products)

        return f"Найдено похожих товаров: {len(similar_products)}\n\n{result_text}", product_ids
    except Exception as e:
        logger.error(f"Ошибка при поиске похожих товаров: {e}", exc_info=True)
        return f"Ошибка при поиске похожих товаров: {str(e)}", []


@tool
async def compare_products(product_ids: List[int]) -> str:
    """Сравнивает несколько товаров по цене, региону, поставщику.

    НАЗНАЧЕНИЕ: Сравнение товаров по цене, региону, поставщику

    ИСПОЛЬЗУЙ ДЛЯ:
    - Когда клиент просит сравнить товары
    - Для выбора лучшего варианта из нескольких товаров
    - Для показа различий между похожими товарами
    - Когда нужно помочь клиенту выбрать между несколькими товарами

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Нужно найти товары (используй vector_search или get_product_by_title)
    - Клиент спрашивает про один товар (используй get_product_by_title)
    - Нужно сравнить более 5 товаров (ограничение: максимум 5 товаров)

    Args:
        product_ids: Список ID товаров для сравнения (минимум 2, максимум 5)

    Returns:
        Отформатированное сравнение товаров с информацией о цене, поставщике, регионе
    """
    if len(product_ids) > 5:
        product_ids = product_ids[:5]
        logger.warning(f"[compare_products] Получено более 5 товаров, сравниваем первые 5")

    if len(product_ids) < 2:
        return "Для сравнения нужно минимум 2 товара."

    try:
        from src.utils import get_supabase_client
        supabase = await get_supabase_client()

        products = []
        for product_id in product_ids:
            try:
                result = (
                    await supabase.table("products")
                    .select("*")
                    .eq("id", product_id)
                    .execute()
                )
                if result.data:
                    products.append(result.data[0])
            except Exception as e:
                logger.warning(f"[compare_products] Ошибка при получении товара ID {product_id}: {e}")
                continue

        if len(products) < 2:
            return "Не удалось найти достаточно товаров для сравнения. Проверьте правильность ID товаров."

        system_vars = await get_all_system_values()
        comparison_lines = ["📊 СРАВНЕНИЕ ТОВАРОВ:\n"]

        for i, product in enumerate(products, 1):
            title = product.get("title", "Не указано")
            supplier = product.get("supplier_name", "Не указано")
            order_price = product.get("order_price_kg")
            region = product.get("from_region", "Не указано")

            final_price = calculate_final_price(
                order_price,
                system_vars,
                supplier_name=supplier
            )

            comparison_lines.append(f"{i}. {title}")
            comparison_lines.append(f"   Поставщик: {supplier}")
            if final_price != "Цена по запросу":
                comparison_lines.append(f"   Цена: {final_price}₽/кг")
            else:
                comparison_lines.append(f"   Цена: {final_price}")
            comparison_lines.append(f"   Регион: {region}")
            comparison_lines.append("")

        return "\n".join(comparison_lines)
    except ValueError as e:
        logger.error(f"[compare_products] Ошибка валидации: {e}", exc_info=True)
        return f"Ошибка валидации: {str(e)}. Проверьте параметры запроса."
    except RuntimeError as e:
        logger.error(f"[compare_products] Ошибка подключения к базе данных: {e}", exc_info=True)
        return "Ошибка подключения к базе данных. Попробуйте позже."
    except Exception as e:
        logger.error(f"[compare_products] Ошибка при сравнении товаров: {e}", exc_info=True)
        return f"Ошибка при сравнении товаров: {str(e)}"


@tool
async def get_products_statistics(stat_type: str = "cheapest", limit: int = 10) -> str:
    """Получает статистику по товарам.

    НАЗНАЧЕНИЕ: Получение статистики по товарам (самые дешевые, самые дорогие)

    ИСПОЛЬЗУЙ ДЛЯ:
    - Когда клиент спрашивает про самые дешевые товары
    - Когда клиент спрашивает про самые дорогие товары
    - Для получения топ товаров по цене
    - Для аналитики по каталогу

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Клиент ищет конкретные товары (используй vector_search)
    - Нужна статистика по другим критериям (пока поддерживается только по цене)

    Args:
        stat_type: Тип статистики:
            - "cheapest" - самые дешевые товары (по умолчанию)
            - "expensive" - самые дорогие товары
        limit: Количество результатов (по умолчанию 10, максимум 20)

    Returns:
        Отформатированная статистика с информацией о товарах
    """
    limit = min(limit, 20)

    try:
        pool = await get_pool()

        async with pool.acquire() as conn:
            if stat_type == "cheapest":
                result = await conn.fetch(
                    """
                    SELECT id, title, supplier_name, from_region, order_price_kg, photo
                    FROM myaso.products
                    WHERE order_price_kg > 0
                    ORDER BY order_price_kg ASC
                    LIMIT $1
                    """,
                    limit
                )
                stat_title = "Самые дешевые товары"
            elif stat_type == "expensive":
                result = await conn.fetch(
                    """
                    SELECT id, title, supplier_name, from_region, order_price_kg, photo
                    FROM myaso.products
                    WHERE order_price_kg > 0
                    ORDER BY order_price_kg DESC
                    LIMIT $1
                    """,
                    limit
                )
                stat_title = "Самые дорогие товары"
            else:
                return f"Неизвестный тип статистики: {stat_type}. Используйте 'cheapest' или 'expensive'."

            products_dict = records_to_json(result)
            result_text, _ = await format_products_list(products_dict)

            return f"{stat_title} ({len(products_dict)} товаров):\n\n{result_text}"
    except ValueError as e:
        logger.error(f"[get_products_statistics] Ошибка валидации: {e}", exc_info=True)
        return f"Ошибка валидации: {str(e)}. Проверьте параметры запроса."
    except RuntimeError as e:
        logger.error(f"[get_products_statistics] Ошибка подключения к базе данных: {e}", exc_info=True)
        return "Ошибка подключения к базе данных. Попробуйте позже."
    except Exception as e:
        logger.error(f"[get_products_statistics] Ошибка при получении статистики: {e}", exc_info=True)
        return f"Ошибка при получении статистики: {str(e)}"


@tool(response_format="content_and_artifact")
async def get_recommendations_based_on_orders(phone: str, limit: int = 10) -> Tuple[str, List[int]]:
    """Получает рекомендации товаров на основе истории заказов клиента.

    НАЗНАЧЕНИЕ: Персонализированные рекомендации товаров на основе истории заказов

    ИСПОЛЬЗУЙ ДЛЯ:
    - Когда клиент просит "что-то похожее на то, что я заказывал"
    - Для персонализированных рекомендаций
    - Когда клиент не знает, что заказать
    - Для предложения товаров на основе предыдущих покупок

    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - У клиента нет истории заказов (используй vector_search или get_random_products)
    - Клиент ищет конкретные товары (используй vector_search или get_product_by_title)

    Args:
        phone: Номер телефона клиента (в формате +7XXXXXXXXXX или 8XXXXXXXXXX)
        limit: Количество рекомендаций (по умолчанию 10, максимум 20)

    Returns:
        Кортеж (текст с рекомендациями, список ID товаров как artifact)
    """
    limit = min(limit, 20)

    try:
        orders = await get_client_orders_db(phone)
        if not orders:
            return "У вас нет истории заказов для рекомендаций. Попробуйте использовать vector_search для поиска товаров.", []

        # Извлекаем названия товаров из заказов
        ordered_titles = [order.title for order in orders if order.title]
        if not ordered_titles:
            return "В истории заказов нет информации о товарах для рекомендаций.", []

        # Используем векторный поиск для каждого заказанного товара
        retriever = SupabaseVectorRetriever()

        all_products = {}
        for title in ordered_titles[:5]:  # Берем максимум 5 последних заказов
            try:
                documents = await retriever.get_relevant_documents(title, k=5)
                for doc in documents:
                    product_id = doc.metadata.get('id')
                    if product_id and product_id not in all_products:
                        all_products[product_id] = doc.metadata
            except Exception as e:
                logger.warning(f"[get_recommendations_based_on_orders] Ошибка при поиске рекомендаций для '{title}': {e}")
                continue

        if not all_products:
            return "Не удалось найти рекомендации на основе ваших заказов.", []

        # Преобразуем в список для форматирования
        products_list = list(all_products.values())[:limit]
        result_text, product_ids = await format_products_list(products_list)

        return f"Рекомендации на основе ваших заказов ({len(products_list)} товаров):\n\n{result_text}", product_ids
    except ValueError as e:
        logger.error(f"[get_recommendations_based_on_orders] Ошибка валидации: {e}", exc_info=True)
        return f"Ошибка валидации: {str(e)}. Проверьте параметры запроса.", []
    except RuntimeError as e:
        logger.error(f"[get_recommendations_based_on_orders] Ошибка подключения к базе данных: {e}", exc_info=True)
        return "Ошибка подключения к базе данных. Попробуйте позже.", []
    except Exception as e:
        logger.error(f"[get_recommendations_based_on_orders] Ошибка при получении рекомендаций: {e}", exc_info=True)
        return f"Ошибка при получении рекомендаций: {str(e)}", []
