from __future__ import annotations

from sqlglot import parse_one
from sqlglot.errors import SqlglotError

def validate_sql_safety(sql_query: str) -> bool:
    """Validate SQL syntax (not permission/safety policy)."""
    if not sql_query or not sql_query.strip():
        return False
    if parse_one is None:
        return True
    try:
        parse_one(sql_query)
    except SqlglotError:
        return False
    return True


def ensure_safe_select(sql_query: str) -> None:
    """Keep compatibility with existing callers; checks SQL is parseable."""
    if not validate_sql_safety(sql_query):
        raise ValueError("Некорректный SQL запрос")


async def validate_sql_conditions(sql_conditions: str) -> None:
    if not sql_conditions or not sql_conditions.strip():
        raise ValueError("SQL условия не могут быть пустыми")
    wrapped = f"SELECT * FROM products WHERE {sql_conditions}"
    if not validate_sql_safety(wrapped):
        raise ValueError("Синтаксическая ошибка SQL условий")
