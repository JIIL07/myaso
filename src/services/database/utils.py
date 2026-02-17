"""Утилиты для работы с базой данных."""

import asyncio
import logging
from typing import Any, Awaitable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_DB_TIMEOUT = 10.0


async def execute_with_timeout(
    coro: Awaitable[T],
    timeout: float = DEFAULT_DB_TIMEOUT,
    operation_name: str = "database operation",
) -> T:
    """Выполняет корутину с timeout для защиты от зависания.

    Args:
        coro: Корутина для выполнения
        timeout: Максимальное время выполнения в секундах (по умолчанию 10.0)
        operation_name: Название операции для логирования

    Returns:
        Результат выполнения корутины

    Raises:
        asyncio.TimeoutError: Если операция превысила timeout
        RuntimeError: Если произошла ошибка при выполнении операции
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(
            f"Timeout при выполнении операции '{operation_name}' ({timeout} секунд)"
        )
        raise RuntimeError(
            f"Database operation '{operation_name}' exceeded timeout of {timeout}s"
        ) from None
    except Exception as e:
        logger.error(
            f"Ошибка при выполнении операции '{operation_name}': {e}",
            exc_info=True,
        )
        raise
