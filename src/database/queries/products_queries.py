"""SQL запросы для работы с товарами."""

from typing import List, Tuple

from src.database import get_pool
from src.models.entities import Product
from src.utils.async_mixin import records_to_json


async def get_random_products(limit: int = 10) -> List[Product]:
    """Получает случайные товары из ассортимента.

    Args:
        limit: Количество товаров для возврата (максимум 20)

    Returns:
        Список моделей Product
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
                    order_price_kg
                FROM myaso.products
                WHERE supplier_name ILIKE '%ООО%КИТ%'
                ORDER BY RANDOM()
                LIMIT $1
                """,
                limit,
            )
            products_dict = records_to_json(result)
            return [Product(**product) for product in products_dict]
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении случайных товаров: {e}") from e


async def get_products_by_sql_conditions(
    sql_conditions: str, limit: int = 50
) -> Tuple[List[Product], bool]:
    """Получает товары по SQL WHERE условиям.

    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE)
        limit: Максимальное количество товаров

    Returns:
        Кортеж (список моделей Product, есть_ли_ещё_товары)
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT
                    id,
                    title,
                    supplier_name,
                    from_region,
                    photo,
                    order_price_kg
                FROM myaso.products
                WHERE {}
                LIMIT $1
            """.format(sql_conditions)
            result = await conn.fetch(query, limit + 1)
            products_dict = records_to_json(result)

            has_more = len(products_dict) > limit
            products = [Product(**product) for product in products_dict[:limit]]

            return (products, has_more)
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товаров по SQL условиям: {e}") from e


async def get_product_by_title(title: str) -> Product | None:
    """Получает товар по названию.

    Args:
        title: Название товара

    Returns:
        Модель Product или None если не найден
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
                return Product(**dict(result))
            return None
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товара по названию: {e}") from e
