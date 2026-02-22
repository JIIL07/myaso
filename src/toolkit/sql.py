from __future__ import annotations

import re

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError


_DANGEROUS_KEYWORDS = (
    "DROP",
    "TRUNCATE",
    "DELETE",
    "INSERT",
    "EXECUTE",
    "EXEC",
    "UPDATE",
    "ALTER",
    "CREATE",
)


def _statement_disallowed(expression: object) -> bool:
    if exp is None:
        return False
    disallowed = tuple(
        cls
        for cls in (
            getattr(exp, "Insert", None),
            getattr(exp, "Update", None),
            getattr(exp, "Delete", None),
            getattr(exp, "TruncateTable", None),
            getattr(exp, "Drop", None),
            getattr(exp, "Alter", None),
            getattr(exp, "Create", None),
            getattr(exp, "Command", None),
        )
        if cls is not None
    )
    if isinstance(expression, disallowed):
        return True
    return any(expression.find(kind) is not None for kind in disallowed)


def _strip_sql_string_literals(sql_text: str) -> str:
    text = re.sub(r"'(?:''|[^'])*'", "''", sql_text)
    text = re.sub(r'"(?:""|[^"])*"', '""', text)
    return text

def validate_sql_safety(sql_query: str) -> bool:
    """Validate SQL safety by blocking dangerous commands."""
    if not sql_query or not sql_query.strip():
        return False
    if parse_one is None:
        sql_upper = _strip_sql_string_literals(sql_query).upper()
        return not any(
            re.search(r"\b" + re.escape(keyword) + r"\b", sql_upper, re.IGNORECASE)
            for keyword in _DANGEROUS_KEYWORDS
        )
    try:
        parsed = parse_one(sql_query)
    except SqlglotError:
        return False
    return not _statement_disallowed(parsed)


def ensure_safe_select(sql_query: str) -> None:
    """Raise when query is invalid or contains dangerous operations."""
    if not validate_sql_safety(sql_query):
        raise ValueError("Обнаружена опасная SQL команда")


async def validate_sql_conditions(sql_conditions: str) -> None:
    if not sql_conditions or not sql_conditions.strip():
        raise ValueError("SQL условия не могут быть пустыми")
    wrapped = f"SELECT * FROM products WHERE {sql_conditions}"
    if not validate_sql_safety(wrapped):
        raise ValueError("Обнаружена опасная SQL команда")
