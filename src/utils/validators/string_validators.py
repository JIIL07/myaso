"""Утилиты для валидации строк."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_not_empty(
    value: Optional[str],
    field_name: str,
    function_name: str = "",
) -> bool:
    """Проверяет, что строковое значение не пустое.

    Args:
        value: Значение для проверки
        field_name: Название поля для логирования (например, "номер телефона", "ID получателя")
        function_name: Название функции для логирования (опционально)

    Returns:
        True если значение валидно (не пустое), False иначе
    """
    if not value or not str(value).strip():
        prefix = f"[{function_name}] " if function_name else ""
        logger.warning(f"{prefix}Пустой {field_name}")
        return False
    return True
