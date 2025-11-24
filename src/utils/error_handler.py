"""Централизованная обработка ошибок."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from src.utils.exceptions import MyasoBaseException

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(
    default_return: Any = None,
    log_error: bool = True,
    reraise: bool = False,
) -> Callable[[F], F]:
    """Декоратор для централизованной обработки ошибок.

    Args:
        default_return: Значение, возвращаемое при ошибке (если reraise=False)
        log_error: Логировать ли ошибку
        reraise: Пробрасывать ли исключение дальше

    Returns:
        Декорированная функция
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except MyasoBaseException as e:
                if log_error:
                    logger.error(
                        f"[{func.__name__}] {e.message}",
                        extra={"details": e.details},
                        exc_info=True,
                    )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                if log_error:
                    logger.error(
                        f"[{func.__name__}] Неожиданная ошибка: {str(e)}",
                        exc_info=True,
                    )
                if reraise:
                    raise
                return default_return

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except MyasoBaseException as e:
                if log_error:
                    logger.error(
                        f"[{func.__name__}] {e.message}",
                        extra={"details": e.details},
                        exc_info=True,
                    )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                if log_error:
                    logger.error(
                        f"[{func.__name__}] Неожиданная ошибка: {str(e)}",
                        exc_info=True,
                    )
                if reraise:
                    raise
                return default_return

        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def log_and_handle_error(
    error: Exception,
    context: str,
    client_phone: str | None = None,
    reraise: bool = False,
) -> None:
    """Логирует и обрабатывает ошибку.

    Args:
        error: Исключение для обработки
        context: Контекст, в котором произошла ошибка
        client_phone: Номер телефона клиента (опционально)
        reraise: Пробрасывать ли исключение дальше
    """
    extra = {"context": context}
    if client_phone:
        extra["client_phone"] = client_phone

    if isinstance(error, MyasoBaseException):
        logger.error(
            f"[{context}] {error.message}",
            extra={**extra, "details": error.details},
            exc_info=True,
        )
    else:
        logger.error(
            f"[{context}] Неожиданная ошибка: {str(error)}",
            extra=extra,
            exc_info=True,
        )

    if reraise:
        raise

