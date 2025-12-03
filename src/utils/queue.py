"""Утилиты для работы с очередью сообщений PGMQ."""

import json
import logging
from typing import Any, Dict, List, Optional

from src.database.database import get_pool

logger = logging.getLogger(__name__)

QUEUE_NAME = "delayed_messages"
DELAY_SECONDS = 900  # 15 минут


async def send_delayed_message(
    client_phone: str, message: str, topic: str, delay: int = DELAY_SECONDS
) -> Optional[int]:
    """Отправляет сообщение в очередь PGMQ с задержкой.

    Args:
        client_phone: Номер телефона клиента
        message: Текст сообщения
        topic: Тема беседы
        delay: Задержка в секундах (по умолчанию 15 минут)

    Returns:
        msg_id сообщения в очереди или None в случае ошибки
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            message_data = {
                "client_phone": client_phone,
                "message": message,
                "topic": topic,
            }

            result = await conn.fetchrow(
                """
                SELECT pgmq.send($1::text, $2::jsonb, $3::integer) as msg_id
                """,
                QUEUE_NAME,
                json.dumps(message_data),
                delay,
            )

            msg_id = result["msg_id"] if result else None
            logger.info(
                f"[queue] Сообщение добавлено в очередь для {client_phone}, msg_id={msg_id}, delay={delay}s"
            )
            return msg_id
    except Exception as e:
        logger.error(
            f"[queue] Ошибка при добавлении сообщения в очередь для {client_phone}: {e}",
            exc_info=True,
        )
        return None


async def read_queue_messages(
    queue_name: str = QUEUE_NAME, vt: int = 30, qty: int = 10
) -> List[Dict[str, Any]]:
    """Читает сообщения из очереди PGMQ.

    Args:
        queue_name: Имя очереди
        vt: Visibility timeout в секундах (сколько времени сообщение будет скрыто)
        qty: Количество сообщений для чтения

    Returns:
        Список сообщений с полями msg_id, read_ct, enqueued_at, vt, message
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetch(
                """
                SELECT * FROM pgmq.read($1::text, $2::integer, $3::integer)
                """,
                queue_name,
                vt,
                qty,
            )

            messages = []
            for row in result:
                messages.append(dict(row))

            return messages
    except Exception as e:
        logger.error(
            f"[queue] Ошибка при чтении сообщений из очереди {queue_name}: {e}",
            exc_info=True,
        )
        return []


async def delete_message(queue_name: str, msg_id: int) -> bool:
    """Удаляет сообщение из очереди по msg_id.

    Args:
        queue_name: Имя очереди
        msg_id: ID сообщения

    Returns:
        True если сообщение удалено, False в случае ошибки
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT pgmq.delete($1::text, $2::bigint) as deleted
                """,
                queue_name,
                msg_id,
            )

            deleted = result["deleted"] if result else False
            if deleted:
                logger.debug(f"[queue] Сообщение {msg_id} удалено из очереди {queue_name}")
            return bool(deleted)
    except Exception as e:
        logger.error(
            f"[queue] Ошибка при удалении сообщения {msg_id} из очереди {queue_name}: {e}",
            exc_info=True,
        )
        return False


async def archive_message(queue_name: str, msg_id: int) -> bool:
    """Архивирует сообщение из очереди по msg_id.

    Args:
        queue_name: Имя очереди
        msg_id: ID сообщения

    Returns:
        True если сообщение архивировано, False в случае ошибки
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT pgmq.archive($1::text, $2::bigint) as archived
                """,
                queue_name,
                msg_id,
            )

            archived = result["archived"] if result else False
            if archived:
                logger.debug(f"[queue] Сообщение {msg_id} архивировано из очереди {queue_name}")
            return bool(archived)
    except Exception as e:
        logger.error(
            f"[queue] Ошибка при архивировании сообщения {msg_id} из очереди {queue_name}: {e}",
            exc_info=True,
        )
        return False

