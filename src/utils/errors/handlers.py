"""Декораторы и утилиты для обработки ошибок."""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_errors(
    context: str = "",
    default_return: Any = None,
    reraise: bool = False,
) -> Callable[[F], F]:
    """Декоратор для логирования ошибок с контекстом.
    
    Args:
        context: Контекст для логирования (например, имя сервиса или функции)
        default_return: Значение, которое возвращается при ошибке (если reraise=False)
        reraise: Если True, исключение пробрасывается дальше после логирования
    
    Returns:
        Декорированная функция
        
    Example:
        @log_errors(context="user_service", default_return=False)
        async def send_notification(user_id: str) -> bool:
            return True
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context_str = f"[{context}] " if context else ""
                logger.error(
                    f"{context_str}Ошибка в {func.__name__}: {e}",
                    exc_info=True,
                )
                if reraise:
                    raise
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context_str = f"[{context}] " if context else ""
                logger.error(
                    f"{context_str}Ошибка в {func.__name__}: {e}",
                    exc_info=True,
                )
                if reraise:
                    raise
                return default_return
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator
