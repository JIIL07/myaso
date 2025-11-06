"""SQL запросы для работы с товарами."""

from typing import List, Dict, Any
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
    sql_conditions: str, limit: int = 15
) -> List[Dict[str, Any]]:
    """Получает товары по SQL WHERE условиям.

    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE)
        limit: Максимальное количество товаров

    Returns:
        Список словарей с данными товаров
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
            result = await conn.fetch(query, limit)
            return records_to_json(result)
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

