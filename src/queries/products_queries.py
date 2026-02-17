"""SQL запросы для работы с товарами."""

import random
from typing import List, Tuple

import asyncpg

from src.entities.product import Product
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout


async def get_random_products(limit: int = 10) -> List[Product]:
    """Получает случайные товары из ассортимента ООО КИТ.

    Args:
        limit: Количество товаров для возврата (максимум 20)

    Returns:
        Список моделей Product
        
    Raises:
        RuntimeError: Если произошла ошибка при выполнении запроса или timeout
    """
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
) -> Tuple[List[Product], bool]:
    """Получает товары по SQL WHERE условиям.
    
    Использует безопасную RPC функцию myaso.safe_product_search для защиты от SQL injection.
    Валидация SQL условий выполняется как на стороне Python, так и в самой RPC функции.

    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE)
        limit: Максимальное количество товаров

    Returns:
        Кортеж (список моделей Product, есть_ли_ещё_товары)
        
    Raises:
        ValueError: Если SQL условия не прошли валидацию
        RuntimeError: Если произошла ошибка при выполнении запроса
    """
    from src.services.database.database import get_pool
    from src.utils.formatters.formatters import records_to_json
    from src.utils.validators.sql_validator import validate_sql_conditions
    
    # Валидация на стороне Python перед вызовом RPC
    await validate_sql_conditions(sql_conditions)
    
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            # Используем безопасную RPC функцию вместо прямого SQL
            # RPC функция дополнительно валидирует условия на стороне БД
            query = "SELECT * FROM myaso.safe_product_search($1::text, $2::int)"
            result = await conn.fetch(query, sql_conditions, limit + 1)
            products_dict = records_to_json(result)

            has_more = len(products_dict) > limit
            products = [Product(**product) for product in products_dict[:limit]]

            return (products, has_more)
    except ValueError:
        # Перебрасываем ValueError от валидации
        raise
    except asyncpg.PostgresSyntaxError as e:
        # Синтаксические ошибки SQL - не retry, это ошибка в запросе
        raise ValueError(f"Синтаксическая ошибка SQL: {e}") from e
    except (asyncpg.PostgresError, asyncpg.InterfaceError) as e:
        # Ошибки подключения или другие ошибки БД - можно retry
        error_msg = str(e)
        # Проверяем, не связана ли ошибка с отсутствием SQL функции
        if "does not exist" in error_msg and "safe_product_search" in error_msg:
            raise RuntimeError(
                f"SQL функция myaso.safe_product_search не найдена в базе данных. "
                f"Необходимо выполнить SQL файл: sql/safe_product_search.sql"
            ) from e
        # Это ошибка подключения или другая ошибка БД
        raise ConnectionError(f"Ошибка подключения к базе данных: {e}") from e
    except Exception as e:
        error_msg = str(e)
        # Проверяем синтаксические ошибки в сообщении (на случай, если они не были пойманы выше)
        if "syntax error" in error_msg.lower() or "syntaxerror" in error_msg.lower():
            raise ValueError(f"Синтаксическая ошибка SQL: {e}") from e
        # Проверяем, не связана ли ошибка с отсутствием SQL функции
        if "does not exist" in error_msg and "safe_product_search" in error_msg:
            raise RuntimeError(
                f"SQL функция myaso.safe_product_search не найдена в базе данных. "
                f"Необходимо выполнить SQL файл: sql/safe_product_search.sql"
            ) from e
        # Остальные ошибки - RuntimeError
        raise RuntimeError(f"Ошибка при получении товаров по SQL условиям: {e}") from e


async def get_product_by_title(title: str) -> Product | None:
    """Получает товар по названию.

    Args:
        title: Название товара

    Returns:
        Модель Product или None если не найден
        
    Raises:
        RuntimeError: Если произошла ошибка при выполнении запроса или timeout
    """
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
