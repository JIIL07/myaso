import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:

    def __init__(self, max_concurrent: int = 1) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current_task: Optional[str] = None
        self._lock = asyncio.Lock()

    async def acquire(self, task_id: Optional[str] = None) -> bool:
        await self._semaphore.acquire()
        async with self._lock:
            self._current_task = task_id
        logger.info(
            "[RateLimiter] Semaphore acquired for task %s, available slots: %s",
            task_id, self._semaphore._value,
        )
        return True

    async def release(self) -> None:
        async with self._lock:
            task_id = self._current_task
            self._current_task = None
        self._semaphore.release()
        logger.info(
            "[RateLimiter] Semaphore released for task %s, available slots: %s",
            task_id, self._semaphore._value,
        )

    def is_available(self) -> bool:
        return self._semaphore._value > 0

    def get_status(self) -> dict:
        try:
            available_slots = getattr(self._semaphore, "_value", 0)
            return {
                "available": self.is_available(),
                "current_task": self._current_task,
                "available_slots": available_slots,
            }
        except Exception as e:
            logger.error("[RateLimiter] Error getting status: %s", e)
            return {
                "available": False,
                "current_task": None,
                "available_slots": 0,
            }
