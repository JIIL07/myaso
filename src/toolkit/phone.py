from __future__ import annotations

import re

from pydantic import TypeAdapter, ValidationError

try:
    from pydantic_extra_types.phone_numbers import PhoneNumber

    _PHONE_ADAPTER = TypeAdapter(PhoneNumber)
except Exception:  # pragma: no cover - fallback for missing optional runtime deps
    PhoneNumber = None
    _PHONE_ADAPTER = None


def _prepare_ru_phone(raw_phone: str) -> str:
    phone = raw_phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if phone.startswith("8") and len(phone) > 1:
        return "+7" + phone[1:]
    if phone.startswith("7") and len(phone) > 1 and not phone.startswith("+"):
        return "+" + phone
    if not phone.startswith("+"):
        if phone.startswith("9") and len(phone) == 10:
            return "+7" + phone
        return "+" + phone
    return phone


def normalize_phone(phone: str) -> str:
    if not phone:
        return phone
    return _prepare_ru_phone(phone)


def validate_phone(phone: str) -> bool:
    if not phone:
        return False
    prepared = normalize_phone(phone)
    if _PHONE_ADAPTER is None:
        return bool(re.match(r"^\+[1-9]\d{9,14}$", prepared))
    try:
        _PHONE_ADAPTER.validate_python(prepared)
    except ValidationError:
        return False
    return True


def normalize_and_validate_phone(phone: str) -> str:
    if not phone or not phone.strip():
        raise ValueError("Номер телефона не может быть пустым")
    prepared = normalize_phone(phone)
    if _PHONE_ADAPTER is None:
        if not validate_phone(prepared):
            raise ValueError("Неверный формат номера телефона")
        return prepared
    try:
        normalized = _PHONE_ADAPTER.validate_python(prepared)
    except ValidationError as exc:
        raise ValueError("Неверный формат номера телефона") from exc
    normalized_value = str(normalized)
    return normalized_value if normalized_value.startswith("+") else prepared


def has_client_phone(client_phone: str | None) -> bool:
    return bool(client_phone and str(client_phone).strip())
