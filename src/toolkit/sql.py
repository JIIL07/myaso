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


def normalize_generated_sql(raw_sql: str) -> str:
    sql_query = (raw_sql or "").strip()
    if not sql_query:
        return ""

    sql_block_pattern = r"```(?:sql)?\s*\n(.*?)```"
    sql_matches = re.findall(sql_block_pattern, sql_query, re.DOTALL | re.IGNORECASE)
    if sql_matches:
        sql_query = sql_matches[0].strip()
    elif sql_query.startswith("```"):
        lines = sql_query.split("\n")
        sql_query = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    for keyword in ("WITH", "SELECT", "WHERE"):
        pos = sql_query.upper().find(keyword)
        if pos > 0:
            sql_query = sql_query[pos:].strip()
            break

    last_semicolon = sql_query.rfind(";")
    if last_semicolon > 0:
        sql_query = sql_query[: last_semicolon + 1].strip()
    else:
        sql_query = sql_query.strip()

    while sql_query.upper().strip().startswith("WHERE"):
        sql_query = sql_query[5:].strip()

    return sql_query


def normalize_runtime_sql(sql_query: str) -> str:
    cleaned = (sql_query or "").strip()
    if cleaned.endswith(";"):
        return cleaned[:-1].strip()
    return cleaned


def is_full_sql_query(sql_query: str) -> bool:
    upper_sql = sql_query.upper()
    return upper_sql.startswith("SELECT") or upper_sql.startswith("WITH")


def add_limit_if_missing(sql_query: str, limit: int) -> str:
    if re.search(r"\bLIMIT\s+\d+\b", sql_query, re.IGNORECASE):
        return sql_query
    return "%s LIMIT %d" % (sql_query, limit)


def format_sql_syntax_error(error_msg: str, sql_query: str, *, is_full_query: bool) -> str:
    query_type = "полный SQL-запрос" if is_full_query else "SQL-условия (WHERE)"
    return (
        "Ошибка синтаксиса SQL: %s\n\nИспользованный %s: %s"
        % (error_msg, query_type, sql_query[:200])
    )
