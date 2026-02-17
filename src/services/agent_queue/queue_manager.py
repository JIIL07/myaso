"""Менеджер локальной очереди задач для агента."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentTask:
    """Задача для обработки агентом."""

    client_phone: str
    message: str
    message_received_time: datetime
    task_type: str  # 'process' или 'init'

    def to_dict(self) -> dict:
        """Преобразует задачу в словарь для сериализации."""
        return {
            "client_phone": self.client_phone,
            "message": self.message,
            "message_received_time": self.message_received_time.isoformat(),
            "task_type": self.task_type,
        }


class QueueManager:
    """Менеджер локальной очереди задач для агента."""

    def __init__(self):
        """Инициализирует менеджер очереди."""
        self._queue: asyncio.Queue[AgentTask] = asyncio.Queue()

    async def add_task(
        self,
        client_phone: str,
        message: str,
        task_type: str,
        message_received_time: Optional[datetime] = None,
    ) -> None:
        """Добавляет задачу в очередь.

        Args:
            client_phone: Номер телефона клиента
            message: Текст сообщения
            task_type: Тип задачи ('process' или 'init')
            message_received_time: Время получения сообщения (если None, используется текущее время)
        """
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
            f"[QueueManager] Задача добавлена в очередь: {task_type} для {client_phone}, "
            f"всего в очереди: {self._queue.qsize()}"
        )

    async def get_task(self) -> Optional[AgentTask]:
        """Получает задачу из очереди (блокирующий вызов).

        Returns:
            Задача из очереди или None, если очередь пуста
        """
        try:
            task = await self._queue.get()
            logger.debug(
                f"[QueueManager] Задача извлечена из очереди: {task.task_type} для {task.client_phone}"
            )
            return task
        except Exception as e:
            logger.error(f"[QueueManager] Ошибка при получении задачи: {e}", exc_info=True)
            return None

    def get_queue_size(self) -> int:
        """Возвращает размер очереди.

        Returns:
            Количество задач в очереди
        """
        return self._queue.qsize()

    def get_all_tasks(self) -> List[dict]:
        """Возвращает все задачи в очереди (без извлечения).

        Returns:
            Список задач в формате словарей
        """
        tasks = []
        # Создаем временную очередь для чтения
        temp_queue = asyncio.Queue()
        queue_size = self._queue.qsize()

        for _ in range(queue_size):
            try:
                task = self._queue.get_nowait()
                tasks.append(task.to_dict())
                temp_queue.put_nowait(task)
            except asyncio.QueueEmpty:
                break

        # Возвращаем задачи обратно в очередь
        for _ in range(len(tasks)):
            try:
                task = temp_queue.get_nowait()
                self._queue.put_nowait(task)
            except asyncio.QueueEmpty:
                break

        return tasks
