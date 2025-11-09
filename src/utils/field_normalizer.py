"""Утилиты для нормализации полей товаров."""


def normalize_field_value(value, field_type: str = "text") -> str:
    """Нормализует значение поля: если значение 0, NULL, или пустое, возвращает 'по запросу'.
    
    Args:
        value: Значение поля из БД (может быть None, 0, '', строка, число)
        field_type: Тип поля - "text" для текстовых, "number" для числовых
    
    Returns:
        Нормализованное значение или "по запросу" если значение отсутствует/пустое/0
    """
    if value is None:
        return "по запросу"
    
    if field_type == "text":
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in ['не указано', 'null', 'none', '']:
                return "по запросу"
            return value_str
        elif isinstance(value, (int, float)) and value == 0:
            return "по запросу"
        return str(value).strip() if str(value).strip() else "по запросу"
    else:
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in ['не указано', 'null', 'none', '']:
                return "по запросу"
            try:
                num_value = float(value_str)
                if num_value == 0:
                    return "по запросу"
                return str(int(num_value)) if num_value.is_integer() else str(num_value)
            except (ValueError, TypeError):
                return "по запросу"
        elif isinstance(value, (int, float)):
            if value == 0:
                return "по запросу"
            return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return "по запросу"

