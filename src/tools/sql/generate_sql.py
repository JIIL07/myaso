"""Tool: generate_sql_from_text — convert natural language to SQL."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.agent.product_agent.policy import get_agent_policy
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.config.settings import settings
from src.services.ai.prompt import escape_prompt_variables, get_prompt
from src.toolkit import ensure_safe_select, normalize_generated_sql, validate_sql_conditions
from src.tools.common._contract import fail_response, ok_response
from src.tools.common._schema import get_products_table_schema

logger = logging.getLogger(__name__)


async def _generate_sql_impl(
    text_conditions: str,
    prompt_name: str | None = None,
) -> str:
    policy = get_agent_policy()
    sql_prompt = None
    if prompt_name:
        sql_prompt = await get_prompt(prompt_name=prompt_name, default_prompt=None)

    try:
        schema_context = await get_products_table_schema()
    except Exception as e:
        raise ValueError("Не удалось получить схему таблиц: %s" % e) from e

    schema_section = "СХЕМА БАЗЫ ДАННЫХ: myaso\n\n%s" % schema_context

    parts = []
    if sql_prompt:
        parts.append(sql_prompt)
    parts.append(schema_section)

    system_prompt = escape_prompt_variables("\n\n".join(parts))

    text2sql_llm = ChatOpenAI(
        model=settings.openrouter.model_id,
        openai_api_key=settings.openrouter.openrouter_api_key,
        openai_api_base=settings.openrouter.base_url,
        temperature=policy.text_to_sql_temperature,
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{text_conditions}")]
    )
    chain = prompt | text2sql_llm

    try:
        result = await chain.ainvoke({"text_conditions": text_conditions})
    except Exception as e:
        logger.error("[generate_sql] Ошибка вызова LLM: %s", e, exc_info=True)
        raise ValueError("Не удалось сгенерировать SQL запрос: %s" % e) from e

    sql_query = normalize_generated_sql(result.content)

    if not sql_query:
        raise ValueError("LLM вернул пустой SQL запрос")

    try:
        ensure_safe_select(sql_query)
    except ValueError as exc:
        logger.error("[generate_sql] Опасная SQL команда: %s", sql_query[:200])
        raise ValueError("Обнаружена опасная SQL команда") from exc

    upper = sql_query.upper().strip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        await validate_sql_conditions(sql_query)

    return sql_query


@tool(response_format="content_and_artifact")
async def generate_sql_from_text(
    text_conditions: str,
    prompt_name: str | None = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> tuple[str, dict[str, Any]]:
    """Генерирует SQL-запрос из текстового описания условий на русском языке.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Нужно перевести запрос клиента в SQL
    - Предыдущий SQL вернул ошибку и нужна перегенерация

    НЕ ИСПОЛЬЗОВАТЬ:
    - Уже есть готовый SQL -> execute_sql_query напрямую
    - Поиск по описанию товара -> vector_search
    - Точное название -> get_product_by_title
    """
    try:
        sql_query = await _generate_sql_impl(
            text_conditions=text_conditions,
            prompt_name=prompt_name,
        )
        return ok_response(
            sql_query,
            artifact={"query": sql_query, "text_conditions": text_conditions},
        )
    except ValueError as e:
        logger.error("[generate_sql] Validation error: %s", e)
        return fail_response(str(e), error_code="sql_generation_error")

