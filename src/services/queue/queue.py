import json
import logging
from typing import Any, Optional

from src.services.database.database import get_pool
from src.constants import QUEUE_NAME, DELAY_SECONDS

logger = logging.getLogger(__name__)


async def send_delayed_message(
    client_phone: str, message: str, delay: int = DELAY_SECONDS
) -> Optional[int]:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            message_data = {
                "client_phone": client_phone,
                "message": message,
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
            logger.info("[Queue] Added msg_id=%s for %s, delay=%ss", msg_id, client_phone, delay)
            return msg_id
    except Exception as e:
        logger.error("[Queue] Error adding for %s: %s", client_phone, e, exc_info=True)
        return None


async def send_delayed_file(
    client_phone: str,
    file_url: str,
    caption: str = "",
    delay: int = DELAY_SECONDS,
) -> Optional[int]:
    """Добавляет файл в очередь для отложенной отправки в Telegram."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            message_data = {
                "client_phone": client_phone,
                "file_url": file_url,
                "caption": caption or "",
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
                "[Queue] Added file msg_id=%s for %s, delay=%ss",
                msg_id, client_phone, delay,
            )
            return msg_id
    except Exception as e:
        logger.error("[Queue] Error adding file for %s: %s", client_phone, e, exc_info=True)
        return None


async def read_queue_messages(
    queue_name: str = QUEUE_NAME, vt: int = 30, qty: int = 10
) -> list[dict[str, Any]]:
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
            return [dict(row) for row in result]
    except Exception as e:
        logger.error("[Queue] Read error from %s: %s", queue_name, e, exc_info=True)
        return []


async def delete_message(queue_name: str, msg_id: int) -> bool:
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
                logger.debug("[Queue] Message %s deleted from %s", msg_id, queue_name)
            return bool(deleted)
    except Exception as e:
        logger.error("[Queue] Delete error %s from %s: %s", msg_id, queue_name, e, exc_info=True)
        return False
