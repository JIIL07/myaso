import logging
import re

from src.services.ai.constants import DANGEROUS_SQL_KEYWORDS

logger = logging.getLogger(__name__)


async def validate_sql_conditions(sql_conditions: str) -> None:
    sql_conditions = sql_conditions.strip()

    if not sql_conditions:
        raise ValueError("SQL условия не могут быть пустыми")

    sql_upper = sql_conditions.upper()

    for keyword in DANGEROUS_SQL_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, sql_upper, re.IGNORECASE):
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")


def validate_sql_safety(sql_query: str) -> bool:
    if not sql_query:
        return False

    sql_upper = sql_query.upper()

    for keyword in DANGEROUS_SQL_KEYWORDS:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, sql_upper, re.IGNORECASE):
            return False

    return True

