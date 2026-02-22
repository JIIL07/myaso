from __future__ import annotations

import random

import asyncpg

from src.entities.product import Product
from src.services.database.database import get_pool
from src.toolkit import records_to_json


async def get_random_products(limit: int = 10) -> list[Product]:
    if limit > 20:
        limit = 20

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, title, supplier_name, from_region, photo, order_price_kg
                FROM myaso.products
                WHERE supplier_name ILIKE $1
                LIMIT $2
                """,
                "%ООО%КИТ%",
                limit * 5,
            )
        products_dict = records_to_json(rows)

        if products_dict:
            products_list = [Product(**product) for product in products_dict]
            random.shuffle(products_list)
            return products_list[:limit]
        return []
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении случайных товаров: {e}") from e


async def get_products_by_sql_conditions(
    sql_conditions: str, limit: int = 50
) -> tuple[list[Product], bool]:
    from src.toolkit import validate_sql_conditions

    await validate_sql_conditions(sql_conditions)

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = "SELECT * FROM safe_product_search($1::text, $2::int)"
            result = await conn.fetch(query, sql_conditions, limit + 1)
            products_dict = records_to_json(result)

            has_more = len(products_dict) > limit
            products = [Product(**product) for product in products_dict[:limit]]

            return (products, has_more)
    except ValueError:
        raise
    except asyncpg.PostgresSyntaxError as e:
        raise ValueError(f"Синтаксическая ошибка SQL: {e}") from e
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as e:
        error_msg = str(e)
        if "does not exist" in error_msg and "safe_product_search" in error_msg:
            raise RuntimeError(
                f"SQL функция safe_product_search не найдена в базе данных. "
                f"Необходимо выполнить SQL файл: sql/safe_product_search.sql"
            ) from e
        raise ConnectionError(f"Ошибка подключения к базе данных: {e}") from e
    except Exception as e:
        error_msg = str(e)
        if "syntax error" in error_msg.lower() or "syntaxerror" in error_msg.lower():
            raise ValueError(f"Синтаксическая ошибка SQL: {e}") from e
        if "does not exist" in error_msg and "safe_product_search" in error_msg:
            raise RuntimeError(
                f"SQL функция safe_product_search не найдена в базе данных. "
                f"Необходимо выполнить SQL файл: sql/safe_product_search.sql"
            ) from e
        raise RuntimeError(f"Ошибка при получении товаров по SQL условиям: {e}") from e


async def get_product_by_title(title: str) -> Product | None:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM myaso.products
                WHERE title = $1
                LIMIT 1
                """,
                title,
            )

        if row:
            return Product(**dict(row))
        return None
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товара по названию: {e}") from e
