"""SQL запросы для работы с товарами."""

from typing import List, Dict, Any, Tuple
import asyncpg
from src.database import get_pool
from src.utils import records_to_json


async def get_random_products(limit: int = 10) -> List[Dict[str, Any]]:
    """Получает случайные товары из ассортимента.

    Args:
        limit: Количество товаров для возврата (максимум 20)

    Returns:
        Список словарей с данными товаров
    """
    if limit > 20:
        limit = 20

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetch(
                """
                SELECT
                    id,
                    title,
                    supplier_name,
                    from_region,
                    photo,
                    order_price_kg,
                    min_order_weight_kg,
                    cooled_or_frozen,
                    ready_made,
                    package_type,
                    discount
                FROM myaso.products
                ORDER BY RANDOM()
                LIMIT $1
                """,
                limit,
            )
            return records_to_json(result)
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении случайных товаров: {e}") from e


async def get_products_by_sql_conditions(
    sql_conditions: str, limit: int = 50
) -> Tuple[List[Dict[str, Any]], bool]:
    """Получает товары по SQL WHERE условиям.

    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE)
        limit: Максимальное количество товаров

    Returns:
        Кортеж (список словарей с данными товаров, есть_ли_ещё_товары)
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = f"""
                SELECT
                    id,
                    title,
                    supplier_name,
                    from_region,
                    photo,
                    order_price_kg,
                    min_order_weight_kg,
                    cooled_or_frozen,
                    ready_made,
                    package_type,
                    discount
                FROM myaso.products
                WHERE {sql_conditions}
                LIMIT $1
            """
            result = await conn.fetch(query, limit + 1)
            products = records_to_json(result)

            has_more = len(products) > limit

            return (products[:limit], has_more)
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товаров по SQL условиям: {e}") from e


async def get_product_by_title(title: str) -> Dict[str, Any] | None:
    """Получает товар по названию.

    Args:
        title: Название товара

    Returns:
        Словарь с данными товара или None если не найден
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT *
                FROM myaso.products
                WHERE title = $1
                LIMIT 1
                """,
                title,
            )
            if result:
                return dict(result)
            return None
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товара по названию: {e}") from e


async def get_products_with_price_drops(limit: int = 50) -> List[Dict[str, Any]]:
    """Получает товары, у которых цены снизились (по таблице price_history).

    Находит товары, у которых текущая цена (order_price_kg) меньше предыдущей цены из истории.

    Args:
        limit: Максимальное количество товаров для возврата

    Returns:
        Список словарей с данными товаров, у которых цены снизились
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetch(
                """
                WITH price_changes AS (
                    SELECT 
                        p.id,
                        p.title,
                        p.supplier_name,
                        p.from_region,
                        p.photo,
                        p.order_price_kg AS current_price,
                        p.min_order_weight_kg,
                        p.cooled_or_frozen,
                        p.ready_made,
                        p.package_type,
                        p.discount,
                        ph_prev.price AS previous_price,
                        ph_prev.date AS previous_date
                    FROM myaso.products p
                    INNER JOIN LATERAL (
                        SELECT price, date
                        FROM myaso.price_history
                        WHERE product = p.title
                          AND suplier_name = p.supplier_name
                          AND date < (
                              SELECT MAX(date)
                              FROM myaso.price_history
                              WHERE product = p.title
                                AND suplier_name = p.supplier_name
                          )
                        ORDER BY date DESC
                        LIMIT 1
                    ) ph_prev ON true
                    WHERE p.order_price_kg < ph_prev.price
                      AND p.photo IS NOT NULL
                      AND p.photo != ''
                    ORDER BY (ph_prev.price - p.order_price_kg) DESC
                    LIMIT $1
                )
                SELECT * FROM price_changes
                """,
                limit,
            )
            return records_to_json(result)
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товаров с пониженными ценами: {e}") from e


async def get_products_by_supplier_with_final_price(
    supplier_name: str, limit: int = 50
) -> List[Dict[str, Any]]:
    """Получает товары поставщика с финальной ценой.

    Args:
        supplier_name: Название поставщика
        limit: Максимальное количество товаров для возврата

    Returns:
        Список словарей с данными товаров, включая финальную цену
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetch(
                """
                SELECT
                    id,
                    title,
                    supplier_name,
                    from_region,
                    photo,
                    order_price_kg,
                    min_order_weight_kg,
                    cooled_or_frozen,
                    ready_made,
                    package_type,
                    discount
                FROM myaso.products
                WHERE supplier_name ILIKE $1
                ORDER BY title
                LIMIT $2
                """,
                f"%{supplier_name}%",
                limit,
            )
            return records_to_json(result)
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товаров поставщика: {e}") from e

