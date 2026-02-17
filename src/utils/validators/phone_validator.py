import logging
import re
from typing import Tuple

from fastapi import HTTPException

logger = logging.getLogger(__name__)

class PhoneValidationError(ValueError):
    def __init__(self, phone: str, message: str = "Invalid phone number"):
        self.phone = phone
        self.message = message
        super().__init__(f"{message}: {phone}")


def normalize_phone(phone: str) -> str:
    if not phone:
        return phone

    phone = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    if phone.startswith("8") and len(phone) > 1:
        phone = "+7" + phone[1:]
    elif phone.startswith("7") and len(phone) > 1 and not phone.startswith("+"):
        phone = "+" + phone
    elif not phone.startswith("+"):
        if phone.startswith("9") and len(phone) == 10:
            phone = "+7" + phone
        else:
            phone = "+" + phone

    return phone


def validate_phone(phone: str) -> bool:
    if not phone:
        logger.debug("Пустой номер телефона")
        return False

    normalized = normalize_phone(phone)
    
    pattern = r"^\+[1-9]\d{9,14}$"
    is_valid = bool(re.match(pattern, normalized))
    
    if not is_valid:
        logger.debug(f"Номер не прошел валидацию: {phone} -> {normalized} (длина: {len(normalized)})")
    
    return is_valid

def get_validated_phone(phone: str) -> Tuple[str, bool]:
    normalized = normalize_phone(phone)
    is_valid = validate_phone(normalized)

    if not is_valid:
        logger.error(f"Невалидный номер телефона: {phone} -> {normalized}")
        raise PhoneValidationError(phone)

    return normalized, is_valid


def validate_phone_dependency(phone: str) -> str:
    try:
        normalized_phone, _ = get_validated_phone(phone)
        return normalized_phone
    except PhoneValidationError as e:
        logger.warning(f"Попытка использовать невалидный номер: {e.phone}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid phone number",
                "message": "Номер телефона должен быть в международном формате",
                "example": "+79123456789",
            },
        )


def validate_client_phone(client_phone: str | None) -> bool:
    return bool(client_phone and client_phone.strip())


