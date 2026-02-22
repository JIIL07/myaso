from __future__ import annotations

import random

import asyncpg

from src.entities.product import Product
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout


async def get_random_products(limit: int = 10) -> list[Product]:
    if limit > 20:
        limit = 20

    try:
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table("products")
            .select("id, title, supplier_name, from_region, photo, order_price_kg")
            .ilike("supplier_name", "%ООО%КИТ%")
            .limit(limit * 5)
            .execute(),
            operation_name=f"get_random_products(limit={limit})",
        )
        
        if result.data:
            products_list = [Product(**product) for product in result.data]
            random.shuffle(products_list)
            return products_list[:limit]
        return []
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении случайных товаров: {e}") from e


async def get_products_by_sql_conditions(
    sql_conditions: str, limit: int = 50
) -> tuple[list[Product], bool]:
    from src.services.database.database import get_pool
    from src.toolkit import records_to_json, validate_sql_conditions

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
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table("products")
            .select("*")
            .eq("title", title)
            .limit(1)
            .execute(),
            operation_name=f"get_product_by_title({title})",
        )
        
        if result.data and len(result.data) > 0:
            return Product(**result.data[0])
        return None
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении товара по названию: {e}") from e
