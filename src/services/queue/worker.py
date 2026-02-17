"""Воркер для обработки отложенных сообщений из очереди PGMQ."""

import asyncio
import json
import logging
from typing import Any, Dict, Set

from src.entities import UserMessageRequest
from src.queries.clients_queries import get_client_send_message
from src.services.ai.conversation import ConversationService
from src.utils.logger.masking import mask_phone
from .queue import delete_message, read_queue_messages

logger = logging.getLogger(__name__)

QUEUE_CHECK_INTERVAL = 30  # Проверять очередь каждые 30 секунд
VISIBILITY_TIMEOUT = 60  # Время видимости сообщения при чтении (60 секунд)
BATCH_SIZE = 10  # Количество сообщений для обработки за раз
GRACEFUL_SHUTDOWN_TIMEOUT = 30.0  # Timeout для graceful shutdown в секундах

# Глобальное состояние для tracking in-flight сообщений
_in_flight_messages: Set[int] = set()
_shutdown_event = asyncio.Event()


async def process_queue_worker() -> None:
    """Основной цикл воркера для обработки сообщений из очереди.

    Периодически проверяет очередь PGMQ и обрабатывает готовые сообщения.
    Останавливается при установке _shutdown_event.
    """
    logger.info("[queue_worker] Воркер очереди запущен")

    while not _shutdown_event.is_set():
        try:
            messages = await read_queue_messages(
                vt=VISIBILITY_TIMEOUT, qty=BATCH_SIZE
            )

            if not messages:
                logger.debug(
                    "[queue_worker] Сообщений в очереди нет (возможно, delay еще не истек)"
                )
                # Используем wait с timeout вместо sleep для возможности прерывания
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=QUEUE_CHECK_INTERVAL)
                    break
                except asyncio.TimeoutError:
                    continue

            logger.info(
                f"[queue_worker] Найдено {len(messages)} сообщений для обработки"
            )

            for msg in messages:
                if _shutdown_event.is_set():
                    logger.info("[queue_worker] Получен сигнал остановки, прекращаем обработку новых сообщений")
                    break
                await process_single_message(msg)

            if not _shutdown_event.is_set():
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("[queue_worker] Воркер очереди остановлен (CancelledError)")
            break
        except Exception as e:
            logger.error(
                f"[queue_worker] Ошибка в цикле обработки очереди: {e}",
                exc_info=True,
            )
            if not _shutdown_event.is_set():
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=QUEUE_CHECK_INTERVAL)
                    break
                except asyncio.TimeoutError:
                    continue
    
    # Ожидаем завершения всех in-flight сообщений
    if _in_flight_messages:
        logger.info(f"[queue_worker] Ожидаем завершения {len(_in_flight_messages)} in-flight сообщений...")
        try:
            await asyncio.wait_for(
                _wait_for_in_flight_completion(),
                timeout=GRACEFUL_SHUTDOWN_TIMEOUT,
            )
            logger.info("[queue_worker] Все in-flight сообщения завершены")
        except asyncio.TimeoutError:
            logger.warning(
                f"[queue_worker] Timeout при ожидании завершения in-flight сообщений. "
                f"Осталось {len(_in_flight_messages)} сообщений"
            )
    
    logger.info("[queue_worker] Воркер очереди полностью остановлен")


async def _wait_for_in_flight_completion() -> None:
    """Ожидает завершения всех in-flight сообщений."""
    while _in_flight_messages:
        await asyncio.sleep(0.5)


async def process_single_message(msg: Dict[str, Any]) -> None:
    """Обрабатывает одно сообщение из очереди.

    Args:
        msg: Сообщение из очереди с полями msg_id, message (JSON), и др.
    """
    msg_id = msg.get("msg_id")
    message_data = msg.get("message")

    if not msg_id or not message_data:
        logger.warning(f"[queue_worker] Некорректное сообщение: {msg}")
        return

    # Добавляем сообщение в tracking in-flight
    _in_flight_messages.add(msg_id)
    
    try:
        if isinstance(message_data, str):
            data = json.loads(message_data)
        else:
            data = message_data

        client_phone = data.get("client_phone")
        message_text = data.get("message")

        if not client_phone or not message_text:
            logger.warning(
                f"[queue_worker] Отсутствуют обязательные поля в сообщении {msg_id}: {data}"
            )
            await delete_message("delayed_messages", msg_id)
            return

        send_message_enabled = await get_client_send_message(client_phone)
        if not send_message_enabled:
            logger.info(
                f"[queue_worker] Пропускаем сообщение {msg_id} для {mask_phone(client_phone)}: send_message=false"
            )
            await delete_message("delayed_messages", msg_id)
            return

        request = UserMessageRequest(
            client_phone=client_phone,
            message=message_text,
        )

        logger.info(
            f"[queue_worker] Обрабатываем сообщение {msg_id} для {mask_phone(client_phone)}"
        )
        
        # Используем ConversationService напрямую вместо импорта из routes
        conversation_service = ConversationService()
        await conversation_service.process_conversation_async(
            client_phone=client_phone,
            message=message_text,
        )

        deleted = await delete_message("delayed_messages", msg_id)
        if deleted:
            logger.info(
                f"[queue_worker] Сообщение {msg_id} для {mask_phone(client_phone)} успешно обработано и удалено из очереди"
            )
        else:
            logger.warning(
                f"[queue_worker] Сообщение {msg_id} для {mask_phone(client_phone)} обработано, но не удалось удалить из очереди"
            )

    except Exception as e:
        logger.error(
            f"[queue_worker] Ошибка при обработке сообщения {msg_id}: {e}",
            exc_info=True,
        )
        try:
            await delete_message("delayed_messages", msg_id)
        except Exception as delete_error:
            logger.error(
                f"[queue_worker] Не удалось удалить сообщение {msg_id} после ошибки: {delete_error}"
            )
    finally:
        # Удаляем сообщение из tracking после завершения обработки
        _in_flight_messages.discard(msg_id)


async def start_queue_worker() -> asyncio.Task:
    """Запускает воркер очереди в фоновом режиме.

    Returns:
        Task воркера для возможности его остановки
    """
    # Сбрасываем shutdown event при запуске
    _shutdown_event.clear()
    _in_flight_messages.clear()
    
    task = asyncio.create_task(process_queue_worker())
    return task


async def graceful_shutdown_worker(timeout: float = GRACEFUL_SHUTDOWN_TIMEOUT) -> None:
    """Выполняет graceful shutdown воркера очереди.

    Устанавливает shutdown event и ждет завершения всех in-flight сообщений.

    Args:
        timeout: Максимальное время ожидания в секундах (по умолчанию 30.0)
    """
    logger.info("[queue_worker] Начинаем graceful shutdown воркера...")
    
    # Устанавливаем shutdown event
    _shutdown_event.set()
    
    # Ожидаем завершения всех in-flight сообщений
    if _in_flight_messages:
        logger.info(f"[queue_worker] Ожидаем завершения {len(_in_flight_messages)} in-flight сообщений...")
        try:
            await asyncio.wait_for(
                _wait_for_in_flight_completion(),
                timeout=timeout,
            )
            logger.info("[queue_worker] Все in-flight сообщения завершены")
        except asyncio.TimeoutError:
            logger.warning(
                f"[queue_worker] Timeout при ожидании завершения in-flight сообщений. "
                f"Осталось {len(_in_flight_messages)} сообщений"
            )
    
    logger.info("[queue_worker] Graceful shutdown завершен")
