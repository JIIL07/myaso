"""Rate limiter для OpenRouter API с использованием семафора."""
import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiter для ограничения одновременных запросов к OpenRouter API."""

    def __init__(self, max_concurrent: int = 1):
        """Инициализирует rate limiter.

        Args:
            max_concurrent: Максимальное количество одновременных запросов (по умолчанию 1)
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current_task: Optional[str] = None
        self._lock = asyncio.Lock()

    async def acquire(self, task_id: Optional[str] = None) -> bool:
        """Захватывает семафор для обработки задачи.

        Args:
            task_id: Идентификатор задачи (опционально, для логирования)

        Returns:
            True если семафор успешно захвачен
        """
        await self._semaphore.acquire()
        async with self._lock:
            self._current_task = task_id
        logger.info(
            f"[RateLimiter] Семафор захвачен для задачи {task_id}, "
            f"доступно слотов: {self._semaphore._value}"
        )
        return True

    async def release(self) -> None:
        """Освобождает семафор после обработки задачи."""
        async with self._lock:
            task_id = self._current_task
            self._current_task = None
        self._semaphore.release()
        logger.info(
            f"[RateLimiter] Семафор освобожден для задачи {task_id}, "
            f"доступно слотов: {self._semaphore._value}"
        )

    def is_available(self) -> bool:
        """Проверяет, доступен ли семафор для захвата.

        Returns:
            True если семафор доступен (есть свободные слоты)
        """
        return self._semaphore._value > 0

    def get_status(self) -> dict:
        """Возвращает статус rate limiter.

        Returns:
            Словарь со статусом: available, current_task, available_slots
        """
        try:
            # Получаем значение семафора через внутренний счетчик
            # Используем _value, но с обработкой ошибок
            available_slots = getattr(self._semaphore, '_value', 0)
            return {
                "available": self.is_available(),
                "current_task": self._current_task,
                "available_slots": available_slots,
            }
        except Exception as e:
            logger.error(f"[RateLimiter] Ошибка получения статуса: {e}")
            return {
                "available": False,
                "current_task": None,
                "available_slots": 0,
            }
