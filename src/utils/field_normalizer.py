"""Утилиты для нормализации полей товаров."""

from src.config.messages_constants import DEFAULT_FIELD_VALUE, EMPTY_VALUES


def normalize_field_value(value, field_type: str = "text") -> str:
    """Нормализует значение поля: если значение 0, NULL, или пустое, возвращает дефолтное значение.
    
    Args:
        value: Значение поля из БД (может быть None, 0, '', строка, число)
        field_type: Тип поля - "text" для текстовых, "number" для числовых
    
    Returns:
        Нормализованное значение или дефолтное значение если значение отсутствует/пустое/0
    """
    if value is None:
        return DEFAULT_FIELD_VALUE
    
    if field_type == "text":
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in EMPTY_VALUES:
                return DEFAULT_FIELD_VALUE
            return value_str
        elif isinstance(value, (int, float)) and value == 0:
            return DEFAULT_FIELD_VALUE
        return str(value).strip() if str(value).strip() else DEFAULT_FIELD_VALUE
    else:
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in EMPTY_VALUES:
                return DEFAULT_FIELD_VALUE
            try:
                num_value = float(value_str)
                if num_value == 0:
                    return DEFAULT_FIELD_VALUE
                return str(int(num_value)) if num_value.is_integer() else str(num_value)
            except (ValueError, TypeError):
                return DEFAULT_FIELD_VALUE
        elif isinstance(value, (int, float)):
            if value == 0:
                return DEFAULT_FIELD_VALUE
            return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return DEFAULT_FIELD_VALUE

