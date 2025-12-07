"""Утилиты для нормализации полей товаров."""

from src.utils.rules import get_rule_as_list, get_rule_as_str


async def normalize_field_value(value, field_type: str = "text") -> str:
    """Нормализует значение поля: если значение 0, NULL, или пустое, возвращает дефолтное значение.
    
    Args:
        value: Значение поля из БД (может быть None, 0, '', строка, число)
        field_type: Тип поля - "text" для текстовых, "number" для числовых
    
    Returns:
        Нормализованное значение или дефолтное значение если значение отсутствует/пустое/0
    """
    # Загружаем правила из БД
    try:
        default_field_value = await get_rule_as_str("DEFAULT_FIELD_VALUE")
    except Exception:
        default_field_value = "по запросу"
    
    try:
        empty_values = await get_rule_as_list("EMPTY_VALUES")
    except Exception:
        empty_values = ["не указано", "null", "none", ""]
    
    if value is None:
        return default_field_value
    
    if field_type == "text":
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in empty_values:
                return default_field_value
            return value_str
        elif isinstance(value, (int, float)) and value == 0:
            return default_field_value
        return str(value).strip() if str(value).strip() else default_field_value
    else:
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in empty_values:
                return default_field_value
            try:
                num_value = float(value_str)
                if num_value == 0:
                    return default_field_value
                return str(int(num_value)) if num_value.is_integer() else str(num_value)
            except (ValueError, TypeError):
                return default_field_value
        elif isinstance(value, (int, float)):
            if value == 0:
                return default_field_value
            return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return default_field_value

