"""Воркер для обработки отложенных сообщений из очереди PGMQ."""

import asyncio
import json
import logging
from typing import Any, Dict

from src.database.queries.clients_queries import get_client_send_message
from src.models import UserMessageRequest
from src.routers.ai_router import process_conversation_background
from src.utils.queue import delete_message, read_queue_messages

logger = logging.getLogger(__name__)

QUEUE_CHECK_INTERVAL = 30  # Проверять очередь каждые 30 секунд
VISIBILITY_TIMEOUT = 60  # Время видимости сообщения при чтении (60 секунд)
BATCH_SIZE = 10  # Количество сообщений для обработки за раз


async def process_queue_worker() -> None:
    """Основной цикл воркера для обработки сообщений из очереди.

    Периодически проверяет очередь PGMQ и обрабатывает готовые сообщения.
    """
    logger.info("[queue_worker] Воркер очереди запущен")

    while True:
        try:
            messages = await read_queue_messages(
                vt=VISIBILITY_TIMEOUT, qty=BATCH_SIZE
            )

            if not messages:
                # Если сообщений нет, логируем для отладки (только периодически)
                logger.debug(
                    "[queue_worker] Сообщений в очереди нет (возможно, delay еще не истек)"
                )
                await asyncio.sleep(QUEUE_CHECK_INTERVAL)
                continue

            logger.info(
                f"[queue_worker] Найдено {len(messages)} сообщений для обработки"
            )

            # Обрабатываем каждое сообщение
            for msg in messages:
                await process_single_message(msg)

            # Небольшая задержка перед следующей проверкой
            await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("[queue_worker] Воркер очереди остановлен")
            break
        except Exception as e:
            logger.error(
                f"[queue_worker] Ошибка в цикле обработки очереди: {e}",
                exc_info=True,
            )
            # При ошибке ждем перед следующей попыткой
            await asyncio.sleep(QUEUE_CHECK_INTERVAL)


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

    try:
        # Парсим JSON данные сообщения
        if isinstance(message_data, str):
            data = json.loads(message_data)
        else:
            data = message_data

        client_phone = data.get("client_phone")
        message_text = data.get("message")
        topic = data.get("topic", "Продать")

        if not client_phone or not message_text:
            logger.warning(
                f"[queue_worker] Отсутствуют обязательные поля в сообщении {msg_id}: {data}"
            )
            await delete_message("delayed_messages", msg_id)
            return

        # Проверяем send_message перед обработкой (на случай если изменилось после добавления в очередь)
        send_message_enabled = await get_client_send_message(client_phone)
        if not send_message_enabled:
            logger.info(
                f"[queue_worker] Пропускаем сообщение {msg_id} для {client_phone}: send_message=false"
            )
            await delete_message("delayed_messages", msg_id)
            return

        # Создаем запрос для обработки
        request = UserMessageRequest(
            client_phone=client_phone,
            message=message_text,
            topic=topic,
        )

        # Обрабатываем сообщение
        logger.info(
            f"[queue_worker] Обрабатываем сообщение {msg_id} для {client_phone}"
        )
        
        await process_conversation_background(request)

        # Удаляем обработанное сообщение из очереди
        deleted = await delete_message("delayed_messages", msg_id)
        if deleted:
            logger.info(
                f"[queue_worker] Сообщение {msg_id} для {client_phone} успешно обработано и удалено из очереди"
            )
        else:
            logger.warning(
                f"[queue_worker] Сообщение {msg_id} для {client_phone} обработано, но не удалось удалить из очереди"
            )

    except Exception as e:
        logger.error(
            f"[queue_worker] Ошибка при обработке сообщения {msg_id}: {e}",
            exc_info=True,
        )
        # В случае ошибки удаляем сообщение, чтобы не обрабатывать его снова
        try:
            await delete_message("delayed_messages", msg_id)
        except Exception as delete_error:
            logger.error(
                f"[queue_worker] Не удалось удалить сообщение {msg_id} после ошибки: {delete_error}"
            )


async def start_queue_worker() -> asyncio.Task:
    """Запускает воркер очереди в фоновом режиме.

    Returns:
        Task воркера для возможности его остановки
    """
    task = asyncio.create_task(process_queue_worker())
    return task

