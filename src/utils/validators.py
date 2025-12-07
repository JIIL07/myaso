"""Валидация SQL условий для защиты от SQL injection.

Предоставляет функцию для безопасной валидации SQL WHERE условий
перед их использованием в запросах к базе данных.
"""

import logging
import re

from src.utils.rules import get_rule_as_list

logger = logging.getLogger(__name__)


async def validate_sql_conditions(sql_conditions: str) -> None:
    """Валидирует SQL WHERE условия на безопасность.

    - Проверяются ТОЛЬКО опасные операции (DROP, TRUNCATE, DELETE, INSERT, EXECUTE, UPDATE, ALTER, CREATE)
    - ВСЕ остальное разрешено: любые колонки, функции, подзапросы, таблицы, схемы, алиасы и т.д.

    Args:
        sql_conditions: SQL WHERE условия для валидации

    Raises:
        ValueError: Если условия не прошли валидацию
    """
    sql_conditions = sql_conditions.strip()

    if not sql_conditions:
        raise ValueError("SQL условия не могут быть пустыми")

    sql_upper = sql_conditions.upper()

    try:
        dangerous_keywords = await get_rule_as_list("DANGEROUS_SQL_KEYWORDS")
    except Exception as e:
        logger.warning(f"[validators] Не удалось загрузить DANGEROUS_SQL_KEYWORDS из БД, используем базовый список: {e}")
        dangerous_keywords = ["DROP", "TRUNCATE", "DELETE", "INSERT", "EXECUTE", "EXEC", "UPDATE", "ALTER", "CREATE"]

    for keyword in dangerous_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper, re.IGNORECASE):
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")
