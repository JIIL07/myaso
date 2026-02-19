from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .queue_manager import QueueManager
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class StatusService:

    def __init__(self, queue_manager: QueueManager, rate_limiter: RateLimiter):
        self.queue_manager = queue_manager
        self.rate_limiter = rate_limiter
        self._current_processing_task: dict[str, Any] | None = None
        self._processing_start_time: datetime | None = None

    def set_current_task(self, task: dict[str, Any] | None, start_time: datetime | None = None) -> None:
        self._current_processing_task = task
        self._processing_start_time = start_time or (datetime.now() if task else None)

    def get_status(self) -> dict[str, Any]:
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
