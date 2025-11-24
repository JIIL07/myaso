"""Утилиты для retry-логики."""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_async(
    func: Callable[..., Any],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Any:
    """Выполняет асинхронную функцию с повторными попытками.
    
    Args:
        func: Асинхронная функция для выполнения
        max_attempts: Максимальное количество попыток
        delay: Начальная задержка между попытками (в секундах)
        backoff: Множитель для задержки (exponential backoff)
        exceptions: Кортеж исключений, при которых нужно повторять
        on_retry: Callback функция, вызываемая при каждой попытке (attempt_num, exception)
    
    Returns:
        Результат выполнения функции
    
    Raises:
        Последнее исключение, если все попытки исчерпаны
    """
    last_exception = None
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                if on_retry:
                    on_retry(attempt, e)
                logger.warning(
                    f"[retry_async] Попытка {attempt}/{max_attempts} не удалась: {e}. "
                    f"Повтор через {current_delay}с"
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"[retry_async] Все {max_attempts} попыток исчерпаны. Последняя ошибка: {e}"
                )
    
    raise last_exception


def retry_sync(
    func: Callable[..., Any],
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """Выполняет синхронную функцию с повторными попытками.
    
    Args:
        func: Синхронная функция для выполнения
        max_attempts: Максимальное количество попыток
        delay: Начальная задержка между попытками (в секундах)
        backoff: Множитель для задержки (exponential backoff)
        exceptions: Кортеж исключений, при которых нужно повторять
    
    Returns:
        Результат выполнения функции
    
    Raises:
        Последнее исключение, если все попытки исчерпаны
    """
    import time
    last_exception = None
    current_delay = delay
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                logger.warning(
                    f"[retry_sync] Попытка {attempt}/{max_attempts} не удалась: {e}. "
                    f"Повтор через {current_delay}с"
                )
                time.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(
                    f"[retry_sync] Все {max_attempts} попыток исчерпаны. Последняя ошибка: {e}"
                )
    
    raise last_exception

