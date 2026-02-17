"""Инструменты для работы с товарами: схема БД, SQL-запросы и text-to-SQL."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import ToolRuntime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.queries.products_queries import (
    get_product_by_title as get_product_by_title_db,
    get_random_products as get_random_products_db,
    get_products_by_sql_conditions,
)
from src.config.settings import settings
from src.services.ai.constants import (
    DANGEROUS_SQL_KEYWORDS,
    DEFAULT_FIELD_VALUE,
    DEFAULT_SQL_LIMIT,
    TEXT_TO_SQL_TEMPERATURE,
)
from src.services.ai.prompt import (
    escape_prompt_variables,
    get_all_system_values,
    get_prompt,
)
from src.utils.formatters.formatters import (
    format_products_list,
    filter_products_by_photo,
    normalize_field_value_sync,
    records_to_json,
)
from src.utils.prices.price_calculator import calculate_final_price
from src.tools.utils import (
    get_require_photo_from_runtime,
    calculate_search_limit,
)
from src.utils.validators import validate_sql_conditions, validate_sql_safety
from src.services.database.database import get_pool
from src.services.database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

ERROR_MESSAGE_DATABASE_NOT_CONFIGURED = "Не настроено подключение к базе данных."

PHOTO_SEARCH_LIMIT_MULTIPLIER = 5

# Кэш для схем таблиц
SCHEMA_CACHE: Dict[str, str] = {}


# ============================================================================
# Схема базы данных
# ============================================================================

async def _fetch_table_schema(table_name: str) -> str:
    """Возвращает описание колонок таблицы из information_schema."""
    if table_name in SCHEMA_CACHE:
        return SCHEMA_CACHE[table_name]

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_schema = 'myaso'
                  AND table_name = $1
                ORDER BY ordinal_position
                """,
                table_name,
            )

        if not rows:
            raise RuntimeError(
                f"Схема таблицы {table_name} не найдена в information_schema"
            )

        lines = []
        for row in rows:
            column = row["column_name"]
            data_type = row["data_type"]
            char_len = row["character_maximum_length"]
            numeric_precision = row["numeric_precision"]
            numeric_scale = row["numeric_scale"]

            if char_len:
                data_type = f"{data_type}({char_len})"
            elif numeric_precision:
                if numeric_scale is not None:
                    data_type = f"{data_type}({numeric_precision},{numeric_scale})"
                else:
                    data_type = f"{data_type}({numeric_precision})"

            nullable = "NULL" if row["is_nullable"] == "YES" else "NOT NULL"
            lines.append(f"- {column} ({data_type}, {nullable})")

        schema_text = "\n".join(lines)
        SCHEMA_CACHE[table_name] = schema_text
        return schema_text
    except Exception as e:
        logger.error(
            "[_fetch_table_schema] Не удалось получить схему таблицы %s из БД: %s",
            table_name,
            e,
        )
        raise


@tool
async def get_database_schema(
    table_name: Optional[str] = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> str:
    """Возвращает схему одной таблицы или всех таблиц в схеме myaso."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if table_name:
                try:
                    schema_text = await _fetch_table_schema(table_name)
                    return f"TABLE: {table_name}\n\nCOLUMNS:\n{schema_text}"
                except RuntimeError:
                    return f"Таблица '{table_name}' не найдена в схеме myaso."
            else:
                table_rows = await conn.fetch(
                    """
                    SELECT DISTINCT table_name
                    FROM information_schema.columns
                    WHERE table_schema = 'myaso'
                    ORDER BY table_name
                    """
                )

                if not table_rows:
                    return "В схеме myaso не найдено таблиц."

                result_parts = []
                for table_row in table_rows:
                    current_table = table_row["table_name"]
                    try:
                        table_schema = await _fetch_table_schema(current_table)
                        result_parts.append(
                            f"TABLE: {current_table}\n\nCOLUMNS:\n{table_schema}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[get_database_schema] Не удалось получить схему таблицы {current_table}: {e}"
                        )
                        result_parts.append(
                            f"TABLE: {current_table}\n\nCOLUMNS:\n(Ошибка получения схемы: {e})"
                        )

                separator = "\n\n" + "=" * 80 + "\n\n"
                return separator.join(result_parts)

    except Exception as e:
        logger.error(
            "[get_database_schema] Ошибка при получении схемы базы данных: %s",
            e,
            exc_info=True,
        )
        return f"Не удалось получить схему базы данных: {e}"


async def get_products_table_schema() -> str:
    """Возвращает схему таблиц products и price_history."""
    products_schema = await _fetch_table_schema("products")
    price_history_schema = await _fetch_table_schema("price_history")
    return f"""
TABLE: products

COLUMNS:
{products_schema}

TABLE: price_history

COLUMNS:
{price_history_schema}
"""


# ============================================================================
# SQL запросы к товарам
# ============================================================================

@tool(response_format="content_and_artifact")
async def get_random_products(
    limit: int = 10,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> Tuple[str, List[int]]:
    """Возвращает случайные товары и их ID, с учётом настроек агента и необходимости фото."""
    require_photo = get_require_photo_from_runtime(runtime)

    try:
        search_limit = calculate_search_limit(limit, require_photo, PHOTO_SEARCH_LIMIT_MULTIPLIER)
        products = await get_random_products_db(search_limit)

        if not products:
            return "Товары не найдены.", []

        products_dict = [product.model_dump() for product in products]

        if require_photo:
            products_dict = filter_products_by_photo(products_dict)
            if not products_dict:
                return "Товары с фотографиями не найдены.", []
            products_dict = products_dict[:limit]

        system_vars = await get_all_system_values()
        result_text, product_ids = await format_products_list(products_dict, system_vars)

        return f"Найдено товаров: {len(products_dict)}\n\n{result_text}", product_ids

    except RuntimeError as e:
        logger.error("[get_random_products] Ошибка подключения к базе данных: %s", e)
        return ERROR_MESSAGE_DATABASE_NOT_CONFIGURED, []
    except Exception as e:
        logger.error(
            "[get_random_products] Ошибка при получении случайных товаров: %s",
            e,
            exc_info=True,
        )
        return f"Ошибка при получении товаров: {str(e)}", []


@tool(response_format="content_and_artifact")
async def get_product_by_title(
    title: str,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> Tuple[str, List[int]]:
    """Ищет товар по точному названию и возвращает описание и ID товара."""
    require_photo = get_require_photo_from_runtime(runtime)

    try:
        product = await get_product_by_title_db(title)
        if not product:
            return f"Товар '{title}' не найден в базе данных.", []

        products_dict = [product.model_dump()]

        if require_photo:
            if not filter_products_by_photo(products_dict):
                return f"Товар '{title}' найден, но не имеет фотографии.", []

        system_vars = await get_all_system_values()
        result_text, product_ids = await format_products_list(products_dict, system_vars)

        return f"Найден товар:\n\n{result_text}", product_ids
    except Exception as e:
        logger.error(
            "[get_product_by_title] Ошибка при поиске товара по названию '%s': %s",
            title,
            e,
            exc_info=True,
        )
        return f"Ошибка при поиске товара: {str(e)}", []


# ============================================================================
# Text2SQL генерация и выполнение
# ============================================================================

def _format_sql_error(error_msg: str, sql_query: str, is_full_query: bool = False) -> str:
    """Формирует человекочитаемое описание SQL-ошибки для агента."""
    query_type = "полный SQL-запрос" if is_full_query else "SQL-условия (WHERE)"
    return (
        f"Ошибка синтаксиса SQL: {error_msg}\n\n"
        f"Использованный {query_type}: {sql_query[:200]}"
    )


async def _generate_sql_from_text_impl(
    text_conditions: str,
    prompt_name: Optional[str] = None,
) -> str:
    """Генерирует SQL (WHERE или полный SELECT) из текстового описания на русском языке."""
    sql_prompt = None
    if prompt_name:
        sql_prompt = await get_prompt(
            prompt_name=prompt_name,
            default_prompt=None,
        )

    try:
        schema_context = await get_products_table_schema()
    except Exception as e:
        raise ValueError(f"Не удалось получить схему таблиц: {e}") from e

    schema_section = f"СХЕМА БАЗЫ ДАННЫХ: myaso\n\n{schema_context}"

    parts = []
    if sql_prompt:
        parts.append(sql_prompt)
    parts.append(schema_section)

    system_prompt = "\n\n".join(parts)
    system_prompt = escape_prompt_variables(system_prompt)

    text2sql_llm = ChatOpenAI(
        model=settings.openrouter.model_id,
        openai_api_key=settings.openrouter.openrouter_api_key,
        openai_api_base=settings.openrouter.base_url,
        temperature=TEXT_TO_SQL_TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{text_conditions}")]
    )
    chain = prompt | text2sql_llm

    try:
        result = await chain.ainvoke({"text_conditions": text_conditions})
    except Exception as e:
        logger.error(
            "[_generate_sql_from_text_impl] Ошибка вызова LLM: %s",
            e,
            exc_info=True,
        )
        raise ValueError(f"Не удалось сгенерировать SQL запрос: {e}") from e

    sql_query = result.content.strip()

    # Извлекаем SQL из markdown блока кода, если он есть
    # Ищем блоки ```sql ... ``` или ``` ... ```
    sql_block_pattern = r"```(?:sql)?\s*\n(.*?)```"
    sql_matches = re.findall(sql_block_pattern, sql_query, re.DOTALL | re.IGNORECASE)
    if sql_matches:
        # Берем первый найденный SQL блок
        sql_query = sql_matches[0].strip()
    elif sql_query.startswith("```"):
        # Если весь ответ в markdown блоке, но без метки sql
        lines = sql_query.split("\n")
        sql_query = "\n".join(
            [line for line in lines if not line.strip().startswith("```")]
        ).strip()
    
    # Удаляем весь текст до первого SQL ключевого слова (SELECT, WITH, WHERE)
    # Это нужно, если LLM вернул объяснения перед SQL
    sql_keywords = ["WITH", "SELECT", "WHERE"]
    for keyword in sql_keywords:
        keyword_pos = sql_query.upper().find(keyword)
        if keyword_pos > 0:
            sql_query = sql_query[keyword_pos:].strip()
            break
    
    # Удаляем текст после последнего ; или после последнего SQL оператора
    # Находим последний значимый SQL оператор
    last_semicolon = sql_query.rfind(";")
    if last_semicolon > 0:
        sql_query = sql_query[:last_semicolon + 1].strip()
    else:
        # Если нет точки с запятой, ищем последний SQL оператор
        # Удаляем все после последнего закрывающего ключевого слова
        sql_query = sql_query.strip()

    while sql_query.upper().strip().startswith("WHERE"):
        sql_query = sql_query[5:].strip()

    if not sql_query:
        raise ValueError("LLM вернул пустой SQL запрос")

    if not validate_sql_safety(sql_query):
        logger.error(
            "Обнаружена опасная SQL команда в запросе: %s",
            sql_query[:200],
        )
        raise ValueError("Обнаружена опасная SQL команда")

    # Проверяем, является ли это полным запросом (SELECT или WITH)
    if not (sql_query.upper().strip().startswith("SELECT") or sql_query.upper().strip().startswith("WITH")):
        await validate_sql_conditions(sql_query)

    return sql_query


@tool(response_format="content_and_artifact")
async def generate_sql_from_text(
    text_conditions: str,
    prompt_name: Optional[str] = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> Tuple[str, Dict]:
    """Генерирует SQL по текстовому описанию условий и возвращает запрос и метаданные."""
    sql_query = await _generate_sql_from_text_impl(
        text_conditions=text_conditions,
        prompt_name=prompt_name,
    )
    artifact = {"query": sql_query, "text_conditions": text_conditions}
    return sql_query, artifact


@tool(response_format="content_and_artifact")
async def execute_sql_query(
    sql_query: str,
    limit: Optional[int] = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> Tuple[str, List[int]]:
    """Выполняет SQL (полный SELECT или условия) и возвращает текст результата и ID товаров."""
    sql_query_clean = sql_query.strip()
    if not sql_query_clean:
        return "SQL запрос пустой.", []

    if sql_query_clean.endswith(";"):
        sql_query_clean = sql_query_clean[:-1].strip()

    if not validate_sql_safety(sql_query_clean):
        return "В запросе обнаружена запрещенная команда", []

    if limit is None:
        limit = DEFAULT_SQL_LIMIT

    require_photo = get_require_photo_from_runtime(runtime)

    logger.debug(
        "[execute_sql_query] Выполнение SQL запроса, require_photo=%s, limit=%s",
        require_photo,
        limit,
    )

    upper_sql = sql_query_clean.upper()
    is_full_query = upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")

    if is_full_query:
        final_query = sql_query_clean
        
        # Добавляем схему myaso к таблицам, если она не указана
        # Заменяем FROM products на FROM myaso.products
        # Заменяем FROM price_history на FROM myaso.price_history
        # Но только если схема не указана
        final_query = re.sub(
            r"\bFROM\s+products\b(?!\.)",
            "FROM myaso.products",
            final_query,
            flags=re.IGNORECASE
        )
        final_query = re.sub(
            r"\bFROM\s+price_history\b(?!\.)",
            "FROM myaso.price_history",
            final_query,
            flags=re.IGNORECASE
        )
        # Также заменяем в JOIN
        final_query = re.sub(
            r"\bJOIN\s+products\b(?!\.)",
            "JOIN myaso.products",
            final_query,
            flags=re.IGNORECASE
        )
        final_query = re.sub(
            r"\bJOIN\s+price_history\b(?!\.)",
            "JOIN myaso.price_history",
            final_query,
            flags=re.IGNORECASE
        )

        upper_sql = final_query.upper()
        if not re.search(r"\bLIMIT\s+\d+\b", upper_sql, re.IGNORECASE):
            final_query = f"{final_query} LIMIT {limit}"

        logger.info("[execute_sql_query] Финальный SQL запрос: %s", final_query)

        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetch(final_query)
        except Exception as e:
            error_msg = str(e)
            logger.error(
                "[execute_sql_query] Ошибка выполнения SQL: %s",
                e,
                exc_info=True,
            )
            # Проверяем, является ли это синтаксической ошибкой
            if "syntax error" in error_msg.lower() or "syntaxerror" in error_msg.lower():
                return _format_sql_error(error_msg, final_query, is_full_query=True), []
            return f"Не удалось выполнить SQL запрос: {e}", []

        if not result:
            return "По указанному запросу ничего не найдено.", []

        json_result = records_to_json(result)
        has_more = False
    else:
        sql_conditions = sql_query_clean

        try:
            await validate_sql_conditions(sql_conditions)
        except ValueError as e:
            logger.error(
                "[execute_sql_query] SQL условия не прошли валидацию: %s. Условия: %s",
                e,
                sql_conditions[:200],
            )
            return f"SQL условия не прошли валидацию: {e}", []

        try:
            search_limit = calculate_search_limit(limit, require_photo, PHOTO_SEARCH_LIMIT_MULTIPLIER)
            products, has_more = await get_products_by_sql_conditions(
                sql_conditions, search_limit
            )
        except ConnectionError as e:
            logger.error(
                "[execute_sql_query] Ошибка подключения к базе данных: %s", e
            )
            return "Не настроено подключение к базе данных.", []
        except ValueError as e:
            # Синтаксическая ошибка SQL - агент должен попробовать исправить запрос
            # Возвращаем детальное сообщение с указанием на необходимость исправления
            error_msg = str(e)
            logger.error(
                "[execute_sql_query] Синтаксическая ошибка SQL: %s. Условия: %s",
                e,
                sql_conditions[:200],
            )
            # Формируем сообщение, которое подскажет агенту исправить SQL через generate_sql_from_text
            return _format_sql_error(error_msg, sql_conditions, is_full_query=False), []
        except RuntimeError as e:
            logger.error(
                "[execute_sql_query] Ошибка при получении товаров: %s", e
            )
            return f"Ошибка при получении товаров: {e}", []
        except Exception as e:
            logger.error(
                "[execute_sql_query] Неожиданная ошибка при получении товаров по SQL условиям: %s",
                e,
                exc_info=True,
            )
            logger.error(
                "[execute_sql_query] SQL условия, которые вызвали ошибку: %s",
                sql_conditions[:200],
            )
            return "Товары по указанным условиям не найдены.", []

        if not products:
            return "Товары по указанным условиям не найдены.", []

        json_result = [product.model_dump() for product in products]

    if require_photo:
        json_result = filter_products_by_photo(json_result)
        if not json_result:
            logger.warning(
                "[execute_sql_query] Не найдено товаров с фото по SQL запросу"
            )
            return "Товары с фотографиями по указанным условиям не найдены.", []

    json_result = json_result[:limit]

    system_vars = await get_all_system_values()
    result_text, product_ids = await format_products_list(json_result, system_vars)

    if is_full_query:
        return f"Найдено строк: {len(json_result)}\n\n{result_text}", product_ids
    else:
        more_text = (
            f"\n\n⚠️ В базе данных есть ещё товары, показываем первые {limit}. "
            "Используйте более конкретные критерии поиска для уточнения."
            if has_more
            else ""
        )
        return (
            f"Найдено товаров: {len(json_result)}{more_text}\n\n{result_text}",
            product_ids,
        )
