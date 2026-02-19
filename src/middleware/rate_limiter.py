"""Rate limiting middleware для FastAPI."""

import logging
from typing import Callable

from fastapi import Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


def setup_rate_limiter(app) -> None:
    """Настраивает rate limiting для приложения.

    Args:
        app: Экземпляр FastAPI приложения
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("[RateLimiter] Настроен")


def get_rate_limit_key(request: Request) -> str:
    """Получает ключ для rate limiting.

    Использует IP адрес клиента или client_phone из запроса, если доступен.

    Args:
        request: Запрос FastAPI

    Returns:
        Строка ключа для rate limiting
    """
    try:
        if hasattr(request.state, "client_phone") and request.state.client_phone:
            return f"client:{request.state.client_phone}"
    except Exception:
        pass

    return get_remote_address(request)
