"""Tool: execute_sql_query — run SQL against the product database."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.policy import get_agent_policy
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import (
    ERROR_MESSAGE_DATABASE_NOT_CONFIGURED,
    SQL_EMPTY_QUERY_MESSAGE,
    SQL_FORBIDDEN_MESSAGE,
)
from src.queries.products_queries import get_products_by_sql_conditions
from src.services.database.database import get_pool
from src.toolkit import (
    add_limit_if_missing,
    ensure_safe_select,
    format_sql_syntax_error,
    is_full_sql_query,
    normalize_runtime_sql,
    records_to_json,
    validate_sql_conditions,
)
from src.tools.common._contract import attach_product_ids, fail_response, ok_response
from src.tools.common._formatting import (
    calculate_search_limit,
    format_and_return_products,
    get_require_photo,
)

logger = logging.getLogger(__name__)
_NOT_FOUND_MESSAGE = "Товары по указанным условиям не найдены."


def _is_sql_syntax_error(error_msg: str) -> bool:
    lower = error_msg.lower()
    return "syntax error" in lower or "syntaxerror" in lower


def _sql_execution_error_response(error: Exception, sql_query: str) -> tuple[str, dict[str, Any]]:
    error_msg = str(error)
    if _is_sql_syntax_error(error_msg):
        return fail_response(
            format_sql_syntax_error(error_msg, sql_query, is_full_query=True),
            error_code="sql_syntax_error",
        )
    return fail_response(
        "Не удалось выполнить SQL запрос: %s" % error,
        error_code="sql_execution_error",
    )


@tool(response_format="content_and_artifact")
async def execute_sql_query(
    sql_query: str,
    limit: int | None = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, dict[str, Any]]:
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
    policy = get_agent_policy()
    sql_query_clean = normalize_runtime_sql(sql_query)
    if not sql_query_clean:
        return fail_response(SQL_EMPTY_QUERY_MESSAGE, error_code="empty_query")

    try:
        ensure_safe_select(sql_query_clean)
    except ValueError:
        return fail_response(SQL_FORBIDDEN_MESSAGE, error_code="forbidden_query")

    limit = policy.clamp_sql_limit(limit)
    require_photo = get_require_photo(runtime)

    logger.debug("[execute_sql_query] require_photo=%s, limit=%s", require_photo, limit)
    is_full_query = is_full_sql_query(sql_query_clean)

    # ------------------------------------------------------------------
    # Full SELECT / WITH query
    # ------------------------------------------------------------------
    if is_full_query:
        final_query = add_limit_if_missing(sql_query_clean, limit)

        logger.info("[execute_sql_query] SQL: %s", final_query)

        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetch(final_query)
        except Exception as e:
            logger.error("[execute_sql_query] Ошибка SQL: %s", e, exc_info=True)
            return _sql_execution_error_response(e, final_query)

        if not result:
            return fail_response("По указанному запросу ничего не найдено.", error_code="not_found")

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
            return fail_response(
                "SQL условия не прошли валидацию: %s" % e,
                error_code="sql_validation_error",
            )

        try:
            search_limit = calculate_search_limit(limit, require_photo)
            products, has_more = await get_products_by_sql_conditions(
                sql_conditions, search_limit
            )
        except ConnectionError as e:
            logger.error("[execute_sql_query] Нет подключения к БД: %s", e)
            return fail_response(
                ERROR_MESSAGE_DATABASE_NOT_CONFIGURED,
                error_code="database_not_configured",
            )
        except ValueError as e:
            logger.error(
                "[execute_sql_query] Синтаксис SQL: %s. Условия: %s",
                e,
                sql_conditions[:200],
            )
            return fail_response(
                format_sql_syntax_error(str(e), sql_conditions, is_full_query=False),
                error_code="sql_syntax_error",
            )
        except RuntimeError as e:
            logger.error("[execute_sql_query] Ошибка получения товаров: %s", e)
            return fail_response(
                "Ошибка при получении товаров: %s" % e,
                error_code="query_runtime_error",
            )
        except Exception as e:
            logger.error(
                "[execute_sql_query] Неожиданная ошибка: %s. Условия: %s",
                e,
                sql_conditions[:200],
                exc_info=True,
            )
            return fail_response(_NOT_FOUND_MESSAGE, error_code="unexpected_error")

        if not products:
            return fail_response(_NOT_FOUND_MESSAGE, error_code="not_found")

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
        return fail_response(text, error_code="not_found_with_photo")

    if is_full_query:
        return ok_response(
            "Найдено строк: %d\n\n%s" % (
                len(product_ids),
                text.split("\n\n", 1)[-1],
            ),
            artifact=attach_product_ids({"limit": limit, "is_full_query": True}, product_ids),
        )

    if has_more:
        text += (
            "\n\n⚠️ В базе данных есть ещё товары, показываем первые %d. "
            "Используйте более конкретные критерии для уточнения." % limit
        )

    return ok_response(
        text,
        artifact=attach_product_ids({"limit": limit, "is_full_query": False}, product_ids),
    )

