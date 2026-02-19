import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:

    client_phone: str
    message: str
    message_received_time: datetime
    task_type: str  # 'process' or 'init'

    def to_dict(self) -> dict:
        return {
            "client_phone": self.client_phone,
            "message": self.message,
            "message_received_time": self.message_received_time.isoformat(),
            "task_type": self.task_type,
        }


class QueueManager:

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentTask] = asyncio.Queue()

    async def add_task(
        self,
        client_phone: str,
        message: str,
        task_type: str,
        message_received_time: Optional[datetime] = None,
    ) -> None:
        if message_received_time is None:
            message_received_time = datetime.now()

        task = AgentTask(
            client_phone=client_phone,
            message=message,
            message_received_time=message_received_time,
            task_type=task_type,
        )

        await self._queue.put(task)
        logger.info(
            "[QueueManager] Task added: %s for %s, queue size: %d",
            task_type, client_phone, self._queue.qsize(),
        )

    async def get_task(self) -> Optional[AgentTask]:
        try:
            task = await self._queue.get()
            logger.debug(
                "[QueueManager] Task retrieved: %s for %s",
                task.task_type, task.client_phone,
            )
            return task
        except Exception as e:
            logger.error("[QueueManager] Error getting task: %s", e, exc_info=True)
            return None

    def get_queue_size(self) -> int:
        return self._queue.qsize()

    def get_all_tasks(self) -> list[dict]:
        tasks = []
        temp_queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        queue_size = self._queue.qsize()

        for _ in range(queue_size):
            try:
                task = self._queue.get_nowait()
                tasks.append(task.to_dict())
                temp_queue.put_nowait(task)
            except asyncio.QueueEmpty:
                break

        for _ in range(len(tasks)):
            try:
                task = temp_queue.get_nowait()
                self._queue.put_nowait(task)
            except asyncio.QueueEmpty:
                break

        return tasks
