def mask_phone(phone: str) -> str:
    """Маскирует номер телефона: +791***89"""
    if not phone or len(str(phone).strip()) < 8:
        return "***"
    phone = str(phone).strip()
    return phone[:4] + "***" + phone[-2:]
