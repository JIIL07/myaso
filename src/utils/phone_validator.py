import re
from typing import Optional


def normalize_phone(phone: str) -> str:
    """Нормализует номер телефона к стандартному формату.

    Удаляет пробелы, приводит к формату с + в начале.
    Обрабатывает случаи, когда номер начинается с пробела и цифры.

    Args:
        phone: Номер телефона в любом формате

    Returns:
        Нормализованный номер телефона (с + в начале)

    Examples:
        >>> normalize_phone("+79991234567")
        '+79991234567'
        >>> normalize_phone(" 79991234567")
        '+79991234567'
        >>> normalize_phone("89991234567")
        '+79991234567'
    """
    if not phone:
        return phone

    phone = phone.strip().replace(" ", "").replace("-", "")

    if phone.startswith(" ") and len(phone) > 1 and phone[1].isdigit():
        phone = "+" + phone[1:]

    if phone.startswith("8") and len(phone) > 1:
        phone = "+7" + phone[1:]

    if not phone.startswith("+"):
        phone = "+" + phone

    return phone


def validate_phone(phone: str) -> bool:
    """Проверяет корректность номера телефона.

    Проверяет, что номер:
    - Начинается с +
    - Содержит только цифры после +
    - Имеет разумную длину (от 10 до 15 цифр)

    Args:
        phone: Номер телефона для проверки

    Returns:
        True если номер валиден, False иначе

    Examples:
        >>> validate_phone("+79991234567")
        True
        >>> validate_phone("+1234567890")
        True
        >>> validate_phone("invalid")
        False
        >>> validate_phone("+123")
        False
    """
    if not phone:
        return False

    normalized = normalize_phone(phone)

    pattern = r"^\+[1-9]\d{9,14}$"
    return bool(re.match(pattern, normalized))


def normalize_and_validate_phone(phone: str) -> tuple[str, bool]:
    """Нормализует и проверяет номер телефона.

    Удобная функция для одновременной нормализации и валидации.

    Args:
        phone: Номер телефона в любом формате

    Returns:
        Кортеж (нормализованный_номер, is_valid)

    Examples:
        >>> normalize_and_validate_phone(" 79991234567")
        ('+79991234567', True)
        >>> normalize_and_validate_phone("invalid")
        ('+invalid', False)
    """
    normalized = normalize_phone(phone)
    is_valid = validate_phone(normalized)
    return normalized, is_valid
