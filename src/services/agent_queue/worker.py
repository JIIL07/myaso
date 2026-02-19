import asyncio
import logging
from datetime import datetime
from typing import Callable, Awaitable, Any, Optional

from .queue_manager import QueueManager, AgentTask
from .rate_limiter import RateLimiter
from .status import StatusService

logger = logging.getLogger(__name__)


class AgentQueueWorker:

    def __init__(
        self,
        queue_manager: QueueManager,
        rate_limiter: RateLimiter,
        status_service: StatusService,
    ) -> None:
        self.queue_manager = queue_manager
        self.rate_limiter = rate_limiter
        self.status_service = status_service
        self._running = False
        self._task_handlers: dict[str, Callable[[str, str, datetime], Awaitable[dict[str, Any]]]] = {}

    def register_handler(
        self,
        task_type: str,
        handler: Callable[[str, str, datetime], Awaitable[dict[str, Any]]],
    ) -> None:
        self._task_handlers[task_type] = handler
        logger.info("[AgentQueueWorker] Handler registered for task type: %s", task_type)

    async def process_task(self, task: AgentTask) -> None:
        task_id = "%s_%s_%s" % (task.task_type, task.client_phone, task.message_received_time.isoformat())

        try:
            await self.rate_limiter.acquire(task_id)

            self.status_service.set_current_task(
                task.to_dict(),
                start_time=datetime.now()
            )

            handler = self._task_handlers.get(task.task_type)
            if not handler:
                logger.error("[AgentQueueWorker] No handler for task type '%s'", task.task_type)
                return

            logger.info(
                "[AgentQueueWorker] Processing task %s: %s for %s",
                task_id, task.task_type, task.client_phone,
            )

            result = await handler(
                task.client_phone,
                task.message,
                task.message_received_time,
            )

            logger.info("[AgentQueueWorker] Task %s completed: %s", task_id, result)

        except Exception as e:
            logger.error("[AgentQueueWorker] Error processing task %s: %s", task_id, e, exc_info=True)
        finally:
            await self.rate_limiter.release()
            self.status_service.set_current_task(None, None)

    async def start(self) -> None:
        if self._running:
            logger.warning("[AgentQueueWorker] Worker already running")
            return

        self._running = True
        logger.info("[AgentQueueWorker] Worker started")

        while self._running:
            try:
                task = await self.queue_manager.get_task()

                if task:
                    await self.process_task(task)
                else:
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("[AgentQueueWorker] Worker received cancel signal")
                break
            except Exception as e:
                logger.error("[AgentQueueWorker] Processing loop error: %s", e, exc_info=True)
                await asyncio.sleep(1)

        self._running = False
        logger.info("[AgentQueueWorker] Worker stopped")

    async def stop(self) -> None:
        self._running = False
        logger.info("[AgentQueueWorker] Stop signal received")

    def is_running(self) -> bool:
        return self._running
