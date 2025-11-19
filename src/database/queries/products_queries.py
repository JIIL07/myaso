"""SQL запросы для работы с товарами."""

from typing import Any, Dict, List, Tuple

from src.database import get_pool
from src.utils import records_to_json


async def get_random_products(limit: int = 10, require_photo: bool = False) -> List[Dict[str, Any]]:
    """Получает случайные товары из ассортимента.

    Args:
        limit: Количество товаров для возврата (максимум 20)
        require_photo: Если True, возвращает только товары с фотографиями

    Returns:
        Список словарей с данными товаров
    """
    if limit > 20:
        limit = 20

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if require_photo:
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
                    WHERE photo IS NOT NULL AND photo != ''
                    ORDER BY RANDOM()
                    LIMIT $1
                    """,
                    limit,
                )
            else:
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
            # Используем форматирование с двойными фигурными скобками для безопасности
            # чтобы избежать проблем, если sql_conditions содержит фигурные скобки
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
