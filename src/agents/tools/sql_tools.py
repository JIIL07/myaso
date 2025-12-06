"""Инструменты для работы с SQL запросами."""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.config.constants import (
    DANGEROUS_SQL_KEYWORDS,
    DEFAULT_SQL_LIMIT,
    TEXT_TO_SQL_TEMPERATURE,
)
from src.config.settings import settings
from src.database import get_pool
from src.database.queries.products_queries import get_products_by_sql_conditions
from src.utils import records_to_json, validate_sql_conditions
from src.utils.prompts import (
    escape_prompt_variables,
    get_prompt,
)
from src.utils.product_formatter import format_products_list, create_product_ids_section

logger = logging.getLogger(__name__)


SCHEMA_CACHE: Dict[str, str] = {}


async def _fetch_table_schema(table_name: str) -> str:
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
            raise RuntimeError(f"Схема таблицы {table_name} не найдена в information_schema")

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
            "[sql_tools] Не удалось получить схему таблицы %s из БД: %s",
            table_name,
            e,
        )
        raise


async def get_products_table_schema() -> str:
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


async def _generate_sql_from_text_impl(
    text_conditions: str,
    topic: Optional[str] = None,
) -> str:
    """Генерирует SQL запрос (WHERE условия или полный SELECT) из текстового описания на русском языке."""
    db_prompt = None
    if topic:
        db_prompt = await get_prompt(topic)

    sql_rules_prompt = await get_prompt("SQL Generation Rules")
    if not sql_rules_prompt:
        logger.warning("[sql_tools] Промпт 'SQL Generation Rules' не найден в БД, используем базовые правила")

    try:
        schema_context = await get_products_table_schema()
    except Exception as e:
        raise ValueError(f"Не удалось получить схему таблиц: {e}") from e

    schema_section = f"СХЕМА БАЗЫ ДАННЫХ: myaso\n\n{schema_context}"

    parts = []
    if db_prompt:
        parts.append(db_prompt)
    if sql_rules_prompt:
        parts.append(sql_rules_prompt)
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
        logger.error("[generate_sql_from_text] Ошибка вызова LLM: %s", e, exc_info=True)
        raise ValueError(f"Не удалось сгенерировать SQL запрос: {e}") from e

    sql_query = result.content.strip()

    # Убираем markdown code blocks если есть
    if sql_query.startswith("```"):
        lines = sql_query.split("\n")
        sql_query = "\n".join([line for line in lines if not line.strip().startswith("```")]).strip()

    # Убираем WHERE в начале если это WHERE условия
    while sql_query.upper().strip().startswith("WHERE"):
        sql_query = sql_query[5:].strip()

    if not sql_query:
        raise ValueError("LLM вернул пустой SQL запрос")

    # Проверка на опасные команды
    sql_upper = sql_query.upper()
    for keyword in DANGEROUS_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", sql_upper):
            logger.error(
                "Обнаружена опасная SQL команда: %s в запросе: %s",
                keyword,
                sql_query[:200],
            )
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")

    # Валидация только для WHERE условий (не для полных SELECT)
    if not sql_query.upper().strip().startswith("SELECT"):
        validate_sql_conditions(sql_query)

    return sql_query

def create_sql_tools():
    """Создает инструменты для работы с SQL.
    
    Returns:
        Список инструментов для работы с SQL
    """
    @tool
    async def generate_sql_from_text(text_conditions: str, topic: Optional[str] = None) -> str:
        """Генерирует SQL запрос (WHERE условия или полный SELECT) из текстового описания на русском языке.

        НАЗНАЧЕНИЕ: Генерирует SQL запрос из текстового описания на русском языке

        АВТОМАТИЧЕСКИ ВЫБИРАЕТ ТИП ЗАПРОСА:
        - Простой запрос (только фильтрация по products) -> генерирует WHERE условия
        - Сложный запрос (нужен JOIN с price_history или подзапросы) -> генерирует полный SELECT запрос

        ИСПОЛЬЗУЙ ДЛЯ:
        - Числовые условия по ЦЕНЕ
        - Числовые условия по ВЕСУ
        - Числовые условия по СКИДКЕ
        - Комбинации числовых условий
        - Поиск всех товаров от поставщика
        - Запросы с JOIN
        - Сложные подзапросы

        Args:
            text_conditions: Текстовое описание условий на русском языке
            topic: Тема диалога для загрузки промпта из БД (опционально)

        Returns:
            SQL запрос (WHERE условия или полный SELECT) для использования в execute_sql_query
        """
        return await _generate_sql_from_text_impl(
            text_conditions=text_conditions,
            topic=topic,
        )

    @tool
    async def execute_sql_query(
        sql_query: str, 
        limit: int = DEFAULT_SQL_LIMIT
    ) -> str:
        """
        Универсальный инструмент для выполнения ЛЮБЫХ SQL SELECT запросов.

        ПРИНИМАЕТ:
        - WHERE условия
        - Полные SELECT запросы
        АВТОМАТИЧЕСКИ ОПРЕДЕЛЯЕТ тип запроса:
        - Если начинается с SELECT -> выполняет как полный запрос
        - Если НЕ начинается с SELECT -> оборачивает в SELECT ... FROM myaso.products WHERE ...

        ВАЖНО:
        1. Используй ТОЛЬКО SELECT запросы!
        2. НЕ используй DROP/DELETE/UPDATE/INSERT/ALTER/CREATE/TRUNCATE/EXECUTE — они запрещены.

    Args:
            sql_query: SQL запрос (WHERE условия или полный SELECT запрос)
        limit: Максимальное количество товаров для возврата (по умолчанию 50)

    Returns:
            Список найденных товаров с ID в секции [PRODUCT_IDS]
        """
        sql_query_clean = sql_query.strip()
        if not sql_query_clean:
            return "SQL запрос пустой."

        if sql_query_clean.endswith(";"):
            sql_query_clean = sql_query_clean[:-1].strip()

        upper_sql = sql_query_clean.upper()
        
        for keyword in DANGEROUS_SQL_KEYWORDS:
            if re.search(rf"\b{keyword}\b", upper_sql):
                return f"В запросе обнаружена запрещенная команда: {keyword}"

        is_full_query = upper_sql.startswith("SELECT")
        
        if is_full_query:
            final_query = sql_query_clean

            upper_sql = final_query.upper()
            if not re.search(r'\bLIMIT\s+\d+\b', upper_sql, re.IGNORECASE):
                final_query = f"{final_query} LIMIT {limit}"

            logger.info(f"[execute_sql_query] Финальный SQL запрос: {final_query}")

            try:
                pool = await get_pool()
                async with pool.acquire() as conn:
                    result = await conn.fetch(final_query)
            except Exception as e:
                logger.error("[execute_sql_query] Ошибка выполнения SQL: %s", e, exc_info=True)
                return f"Не удалось выполнить SQL запрос: {e}"

            if not result:
                return "По указанному запросу ничего не найдено."

            json_result = records_to_json(result)
            has_more = False
        else:
            sql_conditions = sql_query_clean

            try:
                validate_sql_conditions(sql_conditions)
            except ValueError as e:
                logger.error(f"SQL условия не прошли валидацию: {e}. Условия: {sql_conditions[:200]}")
                return f"SQL условия не прошли валидацию: {e}"

            try:
                json_result, has_more = await get_products_by_sql_conditions(sql_conditions, limit)
            except RuntimeError as e:
                logger.error(f"Ошибка подключения к базе данных: {e}")
                return "Не настроено подключение к базе данных."
            except Exception as e:
                logger.error(f"Ошибка при получении товаров по SQL условиям: {e}", exc_info=True)
                logger.error(f"SQL условия, которые вызвали ошибку: {sql_conditions[:200]}")
                return "Товары по указанным условиям не найдены."

            if not json_result:
                return "Товары по указанным условиям не найдены."

        result_text, product_ids = await format_products_list(json_result)
        ids_section = create_product_ids_section(product_ids)

        if is_full_query:
            return f"Найдено строк: {len(json_result)}\n\n{result_text}{ids_section}"
        else:
            more_text = "\n\n⚠️ В базе данных есть ещё товары, показываем первые 50. Используйте более конкретные критерии поиска для уточнения." if has_more else ""
        return f"Найдено товаров: {len(json_result)}{more_text}\n\n{result_text}{ids_section}"

    @tool
    async def get_table_schema() -> str:
        """Получает схему таблиц products и price_history.

        Используй этот инструмент когда нужно узнать структуру таблиц перед генерацией SQL запроса.

        Returns:
            Схема таблиц products и price_history с описанием колонок
        """
        try:
            return await get_products_table_schema()
        except Exception as e:
            logger.error(f"[get_table_schema] Ошибка получения схемы: {e}", exc_info=True)
            return f"Не удалось получить схему таблиц: {e}"

    return [generate_sql_from_text, execute_sql_query, get_table_schema]

