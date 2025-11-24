"""Утилиты для валидации."""

from __future__ import annotations

from src.config.messages_constants import ERROR_MESSAGE_INVALID_PHONE
from src.utils.phone_validator import normalize_phone, validate_phone


def validate_and_normalize_phone(phone: str) -> tuple[str, dict[str, str]]:
    """Валидирует и нормализует номер телефона.

    Args:
        phone: Номер телефона для валидации

    Returns:
        Кортеж (нормализованный_телефон, результат)
        результат содержит:
        - "success": True/False
        - "error": сообщение об ошибке (если success=False)
    """
    normalized_phone = normalize_phone(phone)
    if not validate_phone(normalized_phone):
        return phone, {"success": False, "error": ERROR_MESSAGE_INVALID_PHONE}
    return normalized_phone, {"success": True}

