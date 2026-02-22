"""Tool: get_database_schema — inspect DB table structure."""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.services.database.database import get_pool
from src.tools.common._schema import fetch_table_schema

logger = logging.getLogger(__name__)


@tool
async def get_database_schema(
    table_name: str | None = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> str:
    """Возвращает схему таблиц базы данных (колонки, типы, ограничения).

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Нужно узнать структуру таблиц перед генерацией SQL
    - Нужно проверить названия колонок или типы данных

    НЕ ИСПОЛЬЗОВАТЬ:
    - Структура уже известна из предыдущих вызовов
    - Клиент просто спрашивает о товарах без SQL
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            if table_name:
                try:
                    schema_text = await fetch_table_schema(table_name)
                    return "TABLE: %s\n\nCOLUMNS:\n%s" % (table_name, schema_text)
                except RuntimeError:
                    return "Таблица '%s' не найдена в схеме myaso." % table_name
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
                        table_schema = await fetch_table_schema(current_table)
                        result_parts.append(
                            "TABLE: %s\n\nCOLUMNS:\n%s" % (current_table, table_schema)
                        )
                    except Exception as e:
                        logger.warning(
                            "[get_database_schema] Не удалось получить схему таблицы %s: %s",
                            current_table,
                            e,
                        )
                        result_parts.append(
                            "TABLE: %s\n\nCOLUMNS:\n(Ошибка получения схемы: %s)"
                            % (current_table, e)
                        )

                separator = "\n\n" + "=" * 80 + "\n\n"
                return separator.join(result_parts)

    except Exception as e:
        logger.error(
            "[get_database_schema] Ошибка при получении схемы: %s",
            e,
            exc_info=True,
        )
        return "Не удалось получить схему базы данных: %s" % e

