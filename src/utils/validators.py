"""Валидация SQL условий для защиты от SQL injection.

Предоставляет функцию для безопасной валидации SQL WHERE условий
перед их использованием в запросах к базе данных.
"""

import logging
import re

from src.config.constants import (
    ALLOWED_SQL_COLUMNS,
    ALLOWED_SQL_FUNCTIONS,
    ALLOWED_SQL_OPERATORS,
    DANGEROUS_SQL_KEYWORDS,
    FORBIDDEN_SQL_PATTERNS,
)

logger = logging.getLogger(__name__)


def validate_sql_conditions(sql_conditions: str) -> None:
    """Валидирует SQL WHERE условия на безопасность.

    Проверяет:
    - Используются только разрешенные колонки (whitelist)
    - Используются только разрешенные операторы
    - Нет подзапросов, опасных функций и конструкций

    Args:
        sql_conditions: SQL WHERE условия для валидации

    Raises:
        ValueError: Если условия не прошли валидацию
    """
    sql_conditions = sql_conditions.strip()

    if not sql_conditions:
        raise ValueError("SQL условия не могут быть пустыми")

    sql_upper = sql_conditions.upper()

    for keyword in DANGEROUS_SQL_KEYWORDS:
        if keyword in sql_upper:
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")

    sql_lower = sql_conditions.lower()
    for pattern in FORBIDDEN_SQL_PATTERNS:
        if pattern.lower() in sql_lower:
            raise ValueError(f"Запрещенный паттерн в SQL условиях: {pattern}")

    if re.search(r'\bSELECT\b', sql_upper, re.IGNORECASE):
        raise ValueError("Подзапросы (SELECT) не разрешены в SQL условиях")

    function_pattern = r'\b([A-Z_][A-Z0-9_]*)\s*\('
    functions = re.findall(function_pattern, sql_upper)
    allowed_functions_lower = {f.lower() for f in ALLOWED_SQL_FUNCTIONS}
    for func in functions:
        func_lower = func.lower()
        if func_lower not in allowed_functions_lower:
            raise ValueError(
                f"Использование функции '{func}' не разрешено. "
                f"Разрешенные функции: {', '.join(sorted(ALLOWED_SQL_FUNCTIONS))}"
            )

    column_pattern = r'\b([a-z_][a-z0-9_]*)\b'

    sql_without_strings = re.sub(r"'[^']*'", "'STRING'", sql_conditions)
    sql_without_strings = re.sub(r'"[^"]*"', '"STRING"', sql_without_strings)

    potential_columns = re.findall(column_pattern, sql_without_strings.lower())

    sql_keywords = {
        'where', 'and', 'or', 'not', 'is', 'null', 'in', 'between', 'like', 'ilike',
        'true', 'false', 'current_date', 'current_timestamp', 'now', 'lower', 'upper', 'trim',
        'string',
    }

    used_columns = set()
    for col in potential_columns:
        if col in sql_keywords:
            continue
        if col.replace('_', '').replace('.', '').isdigit():
            continue
        if col not in [op.lower() for op in ALLOWED_SQL_OPERATORS]:
            used_columns.add(col)

    invalid_columns = used_columns - ALLOWED_SQL_COLUMNS
    if invalid_columns:
        raise ValueError(
            f"Использование неразрешенных колонок: {', '.join(sorted(invalid_columns))}. "
            f"Разрешенные колонки: {', '.join(sorted(ALLOWED_SQL_COLUMNS))}"
        )

    if '.' in sql_conditions:
        sql_without_numbers = re.sub(r'\d+\.\d+', 'NUMBER', sql_conditions)
        if '.' in sql_without_numbers:
            raise ValueError("Использование алиасов таблиц или схемы в условиях не разрешено")

    if '"' in sql_conditions:
        raise ValueError("Использование двойных кавычек в SQL условиях не разрешено")

    logger.debug(f"SQL условия прошли валидацию: {sql_conditions[:100]}...")

