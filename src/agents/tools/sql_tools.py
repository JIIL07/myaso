"""Инструменты для работы с SQL запросами."""

from __future__ import annotations

import asyncio
import json
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
from src.utils.field_normalizer import normalize_field_value
from src.utils.price_calculator import calculate_final_price
from src.utils.prompts import (
    escape_prompt_variables,
    get_all_system_values,
    get_prompt,
)

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
    is_init_message: bool = False,
) -> str:
    """Генерирует SQL запрос (WHERE условия или полный SELECT) из текстового описания на русском языке."""
    db_prompt = None
    if topic:
        db_prompt = await get_prompt(topic)

    try:
        schema_context = await get_products_table_schema()
    except Exception as e:
        raise ValueError(f"Не удалось получить схему таблиц: {e}") from e

    schema_context = f"""
    СХЕМА БАЗЫ ДАННЫХ: myaso

    {await get_products_table_schema()}

    ПРАВИЛА ГЕНЕРАЦИИ SQL:

    1. ВЫБОР ТИПА ЗАПРОСА:
       - Если запрос простой (только фильтрация по таблице products) -> генерируй ТОЛЬКО WHERE условия (без SELECT/FROM)
       - Если нужен JOIN с price_history или сложные подзапросы -> генерируй ПОЛНЫЙ SELECT запрос

    2. ДЛЯ WHERE УСЛОВИЙ (простой запрос):
       - Генерируй ТОЛЬКО условия, БЕЗ SELECT/FROM/WHERE
       - Пример: "supplier_name = 'ООО КИТ' AND order_price_kg < 100"
       - Используй ТОЛЬКО колонки из таблицы products
       - НЕ используй алиасы таблиц и схем
       - В подзапросах также НЕ используй алиасы - используй простые имена колонок

    3. ДЛЯ ПОЛНОГО SELECT ЗАПРОСА (сложный запрос с JOIN/подзапросами):
       - Генерируй ПОЛНЫЙ SELECT запрос: SELECT ... FROM myaso.products JOIN myaso.price_history ...
       - Явно указывай схему myaso: myaso.products, myaso.price_history
       - НЕ используй алиасы таблиц (p, ph и т.д.) - обращайся к колонкам напрямую через myaso.products.column
       - Запрос должен возвращать колонки из myaso.products (обязательно id)
       - ВАЖНО: При JOIN с price_history ВСЕГДА используй DISTINCT или EXISTS, так как в price_history может быть несколько записей для одного товара
       - Пример с DISTINCT: "SELECT DISTINCT myaso.products.* FROM myaso.products JOIN myaso.price_history ON myaso.products.title = myaso.price_history.product WHERE ..."
       - Пример с EXISTS: "SELECT myaso.products.* FROM myaso.products WHERE EXISTS (SELECT 1 FROM myaso.price_history WHERE myaso.price_history.product = myaso.products.title AND ...)"

    4. ОБЩИЕ ПРАВИЛА:
       - Используй ТОЛЬКО колонки из списка выше! Никаких других колонок не существует!
       - НЕ используй алиасы таблиц (p, ph, t и т.д.)
       - НЕ используй ключевое слово AS для алиасов"""

    system_prompt = f"{db_prompt}\n\n{schema_context}" if db_prompt else schema_context
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

    if sql_query.startswith("```"):
        lines = sql_query.split("\n")
        sql_query = "\n".join(
                    [line for line in lines if not line.strip().startswith("```")]
        ).strip()

    is_full_query = sql_query.upper().strip().startswith("SELECT")

    if is_full_query:
        products_aliases = re.findall(r'\bFROM\s+myaso\.products\s+(\w+)\b', sql_query, re.IGNORECASE)
        price_history_aliases = re.findall(r'\bJOIN\s+myaso\.price_history\s+(\w+)\b', sql_query, re.IGNORECASE)
        
        for alias in products_aliases:
            sql_query = re.sub(
                rf'\b{alias}\.\*\b',
                'myaso.products.*',
                sql_query,
                flags=re.IGNORECASE
            )
            sql_query = re.sub(
                rf'\b{alias}\.(\w+)\b',
                r'myaso.products.\1',
                sql_query,
                flags=re.IGNORECASE
            )
        
        for alias in price_history_aliases:
            sql_query = re.sub(
                rf'\b{alias}\.(\w+)\b',
                r'myaso.price_history.\1',
                sql_query,
                flags=re.IGNORECASE
            )
        
        sql_query = re.sub(
            r'\bFROM\s+myaso\.(\w+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s+myaso\.)',
            r'FROM myaso.\1',
            sql_query,
            flags=re.IGNORECASE
        )
        sql_query = re.sub(
            r'\bJOIN\s+myaso\.(\w+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s+myaso\.)',
            r'JOIN myaso.\1',
            sql_query,
            flags=re.IGNORECASE
        )
        
        for table in ("products", "price_history"):
            sql_query = re.sub(
                rf'\b(FROM|JOIN)\s+(?!myaso\.){table}\b',
                rf'\1 myaso.{table}',
                sql_query,
                flags=re.IGNORECASE
            )
    else:
        while sql_query.upper().strip().startswith("WHERE"):
            sql_query = sql_query[5:].strip()

        sql_query = re.sub(
            r"\b[a-zA-Z_][a-zA-Z0-9_]*\.([a-zA-Z_][a-zA-Z0-9_]*)\b",
            r"\1",
            sql_query,
        )
        sql_query = re.sub(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+[a-zA-Z_][a-zA-Z0-9_]*\b",
            r"\1",
            sql_query,
            flags=re.IGNORECASE,
        )

    if not sql_query:
        raise ValueError("LLM вернул пустой SQL запрос")

    sql_upper = sql_query.upper()
    for keyword in DANGEROUS_SQL_KEYWORDS:
        if keyword in sql_upper:
            logger.error(
                "Обнаружена опасная SQL команда: %s в запросе: %s",
                keyword,
                sql_query[:200],
            )
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")

    validate_sql_conditions(sql_query)
    return sql_query

def create_sql_tools(is_init_message: bool = False):
    """Создает инструменты для работы с SQL с привязанным is_init_message.
    
    Args:
        is_init_message: Если True, это init_conversation
    
    Returns:
        Список инструментов с модифицированным generate_sql_from_text
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
        - Запросы с JOIN price_history (сравнение цен, история цен)
        - Сложные подзапросы
        - Пустые запросы или init_conversation

        ВАЖНО - НЕ ИСПОЛЬЗУЙ АЛИАСЫ:
        - НЕ используй: products.title, myaso.products.title, p.title, t.column
        - НЕ используй ключевое слово AS для алиасов
        - Для WHERE условий: используй простые имена колонок (title, order_price_kg)
        - Для полных SELECT: используй полные имена (myaso.products.title, myaso.price_history.price)
        - Примеры ПРАВИЛЬНО: 
          * WHERE: "title = 'Грудинка' AND order_price_kg < 100"
          * SELECT: "SELECT myaso.products.* FROM myaso.products JOIN myaso.price_history ..."

        Args:
            text_conditions: Текстовое описание условий на русском языке
            topic: Тема диалога для загрузки промпта из БД (опционально)

        Returns:
            SQL запрос (WHERE условия или полный SELECT) для использования в execute_sql_query
        """
        return await _generate_sql_from_text_impl(
            text_conditions=text_conditions,
            topic=topic,
            is_init_message=is_init_message,
        )

    @tool
    async def execute_sql_query(
        sql_query: str, 
        limit: int = DEFAULT_SQL_LIMIT
    ) -> str:
        """
        Универсальный инструмент для выполнения ЛЮБЫХ SQL SELECT запросов.

        ПРИНИМАЕТ:
        - WHERE условия (например: "supplier_name = 'ООО КИТ' AND order_price_kg < 100")
        - Полные SELECT запросы (например: "SELECT * FROM myaso.products JOIN myaso.price_history ...")

        АВТОМАТИЧЕСКИ ОПРЕДЕЛЯЕТ тип запроса:
        - Если начинается с SELECT -> выполняет как полный запрос
        - Если НЕ начинается с SELECT -> оборачивает в SELECT ... FROM myaso.products WHERE ...

        ВАЖНО:
        1. Используй ТОЛЬКО SELECT запросы!
        2. НЕ используй DROP/DELETE/UPDATE/INSERT/ALTER/CREATE/TRUNCATE/EXECUTE — они запрещены.
        3. Явно указывай схему myaso: например, myaso.products, myaso.price_history.
        4. НЕ используй алиасы таблиц (p, ph и т.д.) — обращайся к колонкам напрямую (myaso.products.title).
        5. Запрос обязан возвращать товары (таблица myaso.products) и иметь колонку id.

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

        products_list = []
        product_ids = []
        system_vars = await get_all_system_values()
        
        for product in json_result:
            product_id = product.get("id")
            if product_id:
                product_ids.append(product_id)

            title = product.get("title", "Не указано")
            supplier = normalize_field_value(product.get("supplier_name"), "text")
            order_price = product.get("order_price_kg")
            region = normalize_field_value(product.get("from_region"), "text")
            
            final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)
            
            product_lines = [f"📦 {title}"]
            product_lines.append(f"   Поставщик: {supplier}")
            if final_price != "Цена по запросу":
                product_lines.append(f"   Цена: {final_price}₽/кг")
            else:
                product_lines.append(f"   Цена: {final_price}")
            product_lines.append(f"   Регион: {region}")
            
            products_list.append("\n".join(product_lines))

        result_text = "\n\n".join(products_list)
        ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
        ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

        if is_full_query:
            return f"Найдено строк: {len(json_result)}\n\n{result_text}{ids_section}"
        else:
            more_text = "\n\n⚠️ В базе данных есть ещё товары, показываем первые 50. Используйте более конкретные критерии поиска для уточнения." if has_more else ""
        return f"Найдено товаров: {len(json_result)}{more_text}\n\n{result_text}{ids_section}"

    return [generate_sql_from_text, execute_sql_query]

