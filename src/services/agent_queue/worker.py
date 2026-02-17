"""Worker для обработки задач из очереди агента."""
import asyncio
import logging
from datetime import datetime
from typing import Callable, Awaitable, Dict, Any, Optional

from .queue_manager import QueueManager, AgentTask
from .rate_limiter import RateLimiter
from .status import StatusService

logger = logging.getLogger(__name__)


class AgentQueueWorker:
    """Worker для автоматической обработки задач из очереди."""

    def __init__(
        self,
        queue_manager: QueueManager,
        rate_limiter: RateLimiter,
        status_service: StatusService,
    ):
        """Инициализирует worker.

        Args:
            queue_manager: Менеджер очереди
            rate_limiter: Rate limiter
            status_service: Сервис статуса
        """
        self.queue_manager = queue_manager
        self.rate_limiter = rate_limiter
        self.status_service = status_service
        self._running = False
        self._task_handlers: Dict[str, Callable[[str, str, datetime], Awaitable[Dict[str, Any]]]] = {}

    def register_handler(
        self,
        task_type: str,
        handler: Callable[[str, str, datetime], Awaitable[Dict[str, Any]]],
    ) -> None:
        """Регистрирует обработчик для типа задачи.

        Args:
            task_type: Тип задачи ('process' или 'init')
            handler: Асинхронная функция-обработчик (client_phone, message, message_received_time) -> result
        """
        self._task_handlers[task_type] = handler
        logger.info(f"[AgentQueueWorker] Зарегистрирован обработчик для типа задачи: {task_type}")

    async def process_task(self, task: AgentTask) -> None:
        """Обрабатывает одну задачу.

        Args:
            task: Задача для обработки
        """
        task_id = f"{task.task_type}_{task.client_phone}_{task.message_received_time.isoformat()}"
        
        try:
            # Захватываем семафор
            await self.rate_limiter.acquire(task_id)
            
            # Устанавливаем текущую задачу в статусе
            self.status_service.set_current_task(
                task.to_dict(),
                start_time=datetime.now()
            )

            # Получаем обработчик для типа задачи
            handler = self._task_handlers.get(task.task_type)
            if not handler:
                logger.error(
                    f"[AgentQueueWorker] Обработчик для типа задачи '{task.task_type}' не найден"
                )
                return

            logger.info(
                f"[AgentQueueWorker] Начинаем обработку задачи {task_id}: "
                f"{task.task_type} для {task.client_phone}"
            )

            # Вызываем обработчик
            result = await handler(
                task.client_phone,
                task.message,
                task.message_received_time,
            )

            logger.info(
                f"[AgentQueueWorker] Задача {task_id} успешно обработана: {result}"
            )

        except Exception as e:
            logger.error(
                f"[AgentQueueWorker] Ошибка при обработке задачи {task_id}: {e}",
                exc_info=True,
            )
        finally:
            # Освобождаем семафор
            await self.rate_limiter.release()
            
            # Очищаем текущую задачу
            self.status_service.set_current_task(None, None)

    async def start(self) -> None:
        """Запускает worker для обработки задач из очереди."""
        if self._running:
            logger.warning("[AgentQueueWorker] Worker уже запущен")
            return

        self._running = True
        logger.info("[AgentQueueWorker] Worker запущен")

        while self._running:
            try:
                # Получаем задачу из очереди (блокирующий вызов)
                task = await self.queue_manager.get_task()
                
                if task:
                    # Обрабатываем задачу
                    await self.process_task(task)
                    
                    # После обработки проверяем очередь на следующую задачу
                    # (цикл продолжит автоматически)
                else:
                    # Если задача None, делаем небольшую паузу
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("[AgentQueueWorker] Worker получил сигнал отмены")
                break
            except Exception as e:
                logger.error(
                    f"[AgentQueueWorker] Ошибка в цикле обработки: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(1)  # Пауза перед повтором

        self._running = False
        logger.info("[AgentQueueWorker] Worker остановлен")

    async def stop(self) -> None:
        """Останавливает worker."""
        self._running = False
        logger.info("[AgentQueueWorker] Получен сигнал остановки worker")

    def is_running(self) -> bool:
        """Проверяет, запущен ли worker.

        Returns:
            True если worker запущен
        """
        return self._running
