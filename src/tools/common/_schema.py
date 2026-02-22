"""Shared database schema helpers for SQL-related tools."""

from __future__ import annotations

import logging

from src.services.database.database import get_pool

logger = logging.getLogger(__name__)

SCHEMA_CACHE: dict[str, str] = {}


async def fetch_table_schema(table_name: str) -> str:
    """Return column descriptions for *table_name* from information_schema (cached)."""
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
                "Схема таблицы %s не найдена в information_schema" % table_name
            )

        lines = []
        for row in rows:
            column = row["column_name"]
            data_type = row["data_type"]
            char_len = row["character_maximum_length"]
            numeric_precision = row["numeric_precision"]
            numeric_scale = row["numeric_scale"]

            if char_len:
                data_type = "%s(%s)" % (data_type, char_len)
            elif numeric_precision:
                if numeric_scale is not None:
                    data_type = "%s(%s,%s)" % (data_type, numeric_precision, numeric_scale)
                else:
                    data_type = "%s(%s)" % (data_type, numeric_precision)

            nullable = "NULL" if row["is_nullable"] == "YES" else "NOT NULL"
            lines.append("- %s (%s, %s)" % (column, data_type, nullable))

        schema_text = "\n".join(lines)
        SCHEMA_CACHE[table_name] = schema_text
        return schema_text
    except Exception as e:
        logger.error(
            "[fetch_table_schema] Не удалось получить схему таблицы %s: %s",
            table_name,
            e,
        )
        raise


async def get_products_table_schema() -> str:
    """Return combined schema for products + price_history tables."""
    products_schema = await fetch_table_schema("products")
    price_history_schema = await fetch_table_schema("price_history")
    return (
        "TABLE: products\n\nCOLUMNS:\n%s\n\n"
        "TABLE: price_history\n\nCOLUMNS:\n%s"
        % (products_schema, price_history_schema)
    )

