"""Tool: execute_sql_query — run SQL against the product database."""

from __future__ import annotations

import logging
import re

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import DEFAULT_SQL_LIMIT, PHOTO_SEARCH_LIMIT_MULTIPLIER
from src.queries.products_queries import get_products_by_sql_conditions
from src.services.database.database import get_pool
from src.toolkit import ensure_safe_select, records_to_json, validate_sql_conditions
from src.tools._formatting import format_and_return_products

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def execute_sql_query(
    sql_query: str,
    limit: int | None = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, list[int]]:
    """Выполняет SQL-запрос и возвращает найденные товары.

    Принимает WHERE-условия или полный SELECT-запрос.
    Права на операции контролируются ролью базы данных.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Есть готовый SQL (из generate_sql_from_text или составленный вручную)
    - Поиск по конкретным числовым критериям: цена, регион, поставщик

    НЕ ИСПОЛЬЗОВАТЬ:
    - Поиск по описанию -> vector_search
    - Точное название -> get_product_by_title
    - Нет конкретных условий -> get_random_products
    """
    sql_query_clean = sql_query.strip()
    if not sql_query_clean:
        return "SQL запрос пустой.", []

    if sql_query_clean.endswith(";"):
        sql_query_clean = sql_query_clean[:-1].strip()

    try:
        ensure_safe_select(sql_query_clean)
    except ValueError:
        return "Некорректный SQL запрос", []

    if limit is None:
        limit = DEFAULT_SQL_LIMIT

    require_photo = bool(runtime and runtime.state.get("require_photo", False))

    logger.debug(
        "[execute_sql_query] require_photo=%s, limit=%s",
        require_photo,
        limit,
    )

    upper_sql = sql_query_clean.upper()
    is_full_query = upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")

    # ------------------------------------------------------------------
    # Full SELECT / WITH query
    # ------------------------------------------------------------------
    if is_full_query:
        final_query = sql_query_clean

        if not re.search(r"\bLIMIT\s+\d+\b", final_query, re.IGNORECASE):
            final_query = "%s LIMIT %d" % (final_query, limit)

        logger.info("[execute_sql_query] SQL: %s", final_query)

        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetch(final_query)
        except Exception as e:
            error_msg = str(e)
            logger.error("[execute_sql_query] Ошибка SQL: %s", e, exc_info=True)
            if "syntax error" in error_msg.lower() or "syntaxerror" in error_msg.lower():
                return (
                    "Ошибка синтаксиса SQL: %s\n\nИспользованный полный SQL-запрос: %s"
                    % (error_msg, final_query[:200]),
                    [],
                )
            return "Не удалось выполнить SQL запрос: %s" % e, []

        if not result:
            return "По указанному запросу ничего не найдено.", []

        json_result = records_to_json(result)
        has_more = False

    # ------------------------------------------------------------------
    # WHERE-conditions only
    # ------------------------------------------------------------------
    else:
        sql_conditions = sql_query_clean

        try:
            await validate_sql_conditions(sql_conditions)
        except ValueError as e:
            logger.error(
                "[execute_sql_query] Валидация: %s. Условия: %s",
                e,
                sql_conditions[:200],
            )
            return "SQL условия не прошли валидацию: %s" % e, []

        try:
            search_limit = (limit * PHOTO_SEARCH_LIMIT_MULTIPLIER) if require_photo else limit
            products, has_more = await get_products_by_sql_conditions(
                sql_conditions, search_limit
            )
        except ConnectionError as e:
            logger.error("[execute_sql_query] Нет подключения к БД: %s", e)
            return "Не настроено подключение к базе данных.", []
        except ValueError as e:
            logger.error(
                "[execute_sql_query] Синтаксис SQL: %s. Условия: %s",
                e,
                sql_conditions[:200],
            )
            return (
                "Ошибка синтаксиса SQL: %s\n\nИспользованные SQL-условия (WHERE): %s"
                % (str(e), sql_conditions[:200]),
                [],
            )
        except RuntimeError as e:
            logger.error("[execute_sql_query] Ошибка получения товаров: %s", e)
            return "Ошибка при получении товаров: %s" % e, []
        except Exception as e:
            logger.error(
                "[execute_sql_query] Неожиданная ошибка: %s. Условия: %s",
                e,
                sql_conditions[:200],
                exc_info=True,
            )
            return "Товары по указанным условиям не найдены.", []

        if not products:
            return "Товары по указанным условиям не найдены.", []

        json_result = [product.model_dump() for product in products]

    # ------------------------------------------------------------------
    # Common: photo filter + format
    # ------------------------------------------------------------------
    text, product_ids = await format_and_return_products(
        json_result,
        require_photo=require_photo,
        limit=limit,
        no_photo_message="Товары с фотографиями по указанным условиям не найдены.",
    )

    if not product_ids:
        return text, product_ids

    if is_full_query:
        return "Найдено строк: %d\n\n%s" % (
            len(product_ids),
            text.split("\n\n", 1)[-1],
        ), product_ids

    if has_more:
        text += (
            "\n\n⚠️ В базе данных есть ещё товары, показываем первые %d. "
            "Используйте более конкретные критерии для уточнения." % limit
        )

    return text, product_ids
