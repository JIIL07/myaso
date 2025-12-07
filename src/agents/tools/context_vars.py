"""Контекстные переменные для инструментов."""

from contextvars import ContextVar

client_phone_context: ContextVar[str] = ContextVar('client_phone', default='')


def get_client_phone() -> str:
    """Получает client_phone из контекста выполнения.
    
    Returns:
        Номер телефона клиента
        
    Raises:
        ValueError: Если client_phone не установлен в контексте
    """
    phone = client_phone_context.get()
    if not phone:
        raise ValueError("client_phone не установлен в контексте")
    return phone

