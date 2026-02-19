import asyncio
import logging
from typing import Awaitable, TypeVar

from src.constants import DEFAULT_DB_TIMEOUT

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def execute_with_timeout(
    coro: Awaitable[T],
    timeout: float = DEFAULT_DB_TIMEOUT,
    operation_name: str = "database operation",
) -> T:
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("[DB] Timeout %s (%ss)", operation_name, timeout)
        raise RuntimeError("[DB] Timeout %s (%ss)" % (operation_name, timeout)) from None
    except Exception as e:
        logger.error("[DB] Error %s: %s", operation_name, e, exc_info=True)
        raise
