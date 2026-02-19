import re
from src.constants import DANGEROUS_SQL_KEYWORDS


async def validate_sql_conditions(sql_conditions: str) -> None:
    if not sql_conditions.strip():
        raise ValueError("SQL условия не могут быть пустыми")
    sql_upper = sql_conditions.upper()
    for keyword in DANGEROUS_SQL_KEYWORDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", sql_upper, re.IGNORECASE):
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")


def validate_sql_safety(sql_query: str) -> bool:
    if not sql_query:
        return False
    sql_upper = sql_query.upper()
    return not any(
        re.search(r"\b" + re.escape(kw) + r"\b", sql_upper, re.IGNORECASE)
        for kw in DANGEROUS_SQL_KEYWORDS
    )
