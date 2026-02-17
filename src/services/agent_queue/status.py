"""Сервис для получения статуса очереди и агента."""
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from .queue_manager import QueueManager
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class StatusService:
    """Сервис для получения статуса очереди и агента."""

    def __init__(self, queue_manager: QueueManager, rate_limiter: RateLimiter):
        """Инициализирует сервис статуса.

        Args:
            queue_manager: Менеджер очереди
            rate_limiter: Rate limiter
        """
        self.queue_manager = queue_manager
        self.rate_limiter = rate_limiter
        self._current_processing_task: Optional[Dict[str, Any]] = None
        self._processing_start_time: Optional[datetime] = None

    def set_current_task(self, task: Optional[Dict[str, Any]], start_time: Optional[datetime] = None) -> None:
        """Устанавливает текущую обрабатываемую задачу.

        Args:
            task: Информация о задаче или None
            start_time: Время начала обработки
        """
        self._current_processing_task = task
        self._processing_start_time = start_time or (datetime.now() if task else None)

    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус очереди и агента.

        Returns:
            Словарь со статусом:
            - agent_status: 'free' или 'busy'
            - queue_size: количество задач в очереди
            - queue_tasks: список задач в очереди
            - current_task: текущая обрабатываемая задача (если есть)
        """
        rate_limiter_status = self.rate_limiter.get_status()
        agent_status = "free" if rate_limiter_status["available"] else "busy"

        queue_tasks = self.queue_manager.get_all_tasks()
        queue_size = len(queue_tasks)

        current_task_info = None
        if self._current_processing_task:
            current_task_info = {
                "client_phone": self._current_processing_task.get("client_phone"),
                "task_type": self._current_processing_task.get("task_type"),
                "started_at": (
                    self._processing_start_time.isoformat()
                    if self._processing_start_time
                    else None
                ),
            }

        return {
            "agent_status": agent_status,
            "queue_size": queue_size,
            "queue_tasks": queue_tasks,
            "current_task": current_task_info,
            "rate_limiter": rate_limiter_status,
        }
