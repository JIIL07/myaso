import asyncio
import json
import logging
from typing import Any

from src.entities import UserMessageRequest
from src.queries.clients_queries import get_client_send_message
from src.utils.logger.masking import mask_phone
from src.constants import QUEUE_CHECK_INTERVAL, VISIBILITY_TIMEOUT, BATCH_SIZE, GRACEFUL_SHUTDOWN_TIMEOUT
from .queue import delete_message, read_queue_messages

logger = logging.getLogger(__name__)

_in_flight_messages: set[int] = set()
_shutdown_event = asyncio.Event()


async def process_queue_worker() -> None:
    logger.info("[QueueWorker] Queue worker started")

    while not _shutdown_event.is_set():
        try:
            messages = await read_queue_messages(
                vt=VISIBILITY_TIMEOUT, qty=BATCH_SIZE
            )

            if not messages:
                logger.debug("[QueueWorker] Queue empty")
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=QUEUE_CHECK_INTERVAL)
                    break
                except asyncio.TimeoutError:
                    continue

            logger.info("[QueueWorker] Found %d messages to process", len(messages))

            for msg in messages:
                if _shutdown_event.is_set():
                    logger.info("[QueueWorker] Shutdown signal received, stopping new message processing")
                    break
                await process_single_message(msg)

            if not _shutdown_event.is_set():
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("[QueueWorker] Worker stopped (CancelledError)")
            break
        except Exception as e:
            logger.error("[QueueWorker] Queue processing loop error: %s", e, exc_info=True)
            if not _shutdown_event.is_set():
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=QUEUE_CHECK_INTERVAL)
                    break
                except asyncio.TimeoutError:
                    continue

    if _in_flight_messages:
        logger.info("[QueueWorker] Waiting for %d in-flight messages", len(_in_flight_messages))
        try:
            await asyncio.wait_for(
                _wait_for_in_flight_completion(),
                timeout=GRACEFUL_SHUTDOWN_TIMEOUT,
            )
            logger.info("[QueueWorker] All in-flight messages completed")
        except asyncio.TimeoutError:
            logger.warning(
                "[QueueWorker] Timeout waiting for in-flight messages. %d remaining",
                len(_in_flight_messages),
            )

    logger.info("[QueueWorker] Worker stopped")


async def _wait_for_in_flight_completion() -> None:
    while _in_flight_messages:
        await asyncio.sleep(0.5)


async def process_single_message(msg: dict[str, Any]) -> None:
    msg_id = msg.get("msg_id")
    message_data = msg.get("message")

    if not msg_id or not message_data:
        logger.warning("[QueueWorker] Invalid message: %s", msg)
        return

    _in_flight_messages.add(msg_id)

    try:
        if isinstance(message_data, str):
            data = json.loads(message_data)
        else:
            data = message_data

        client_phone = data.get("client_phone")
        message_text = data.get("message")

        if not client_phone or not message_text:
            logger.warning("[QueueWorker] Missing required fields in message %s", msg_id)
            await delete_message("delayed_messages", msg_id)
            return

        send_message_enabled = await get_client_send_message(client_phone)
        if not send_message_enabled:
            logger.info(
                "[QueueWorker] Skipping %s for %s: send_message=false",
                msg_id, mask_phone(client_phone),
            )
            await delete_message("delayed_messages", msg_id)
            return

        logger.info("[QueueWorker] Processing %s for %s", msg_id, mask_phone(client_phone))

        from src.services.ai.conversation import ConversationService
        conversation_service = ConversationService.instance()
        await conversation_service.process_conversation_async(
            client_phone=client_phone,
            message=message_text,
        )

        deleted = await delete_message("delayed_messages", msg_id)
        if deleted:
            logger.info("[QueueWorker] Message %s processed for %s", msg_id, mask_phone(client_phone))
        else:
            logger.warning("[QueueWorker] Message %s processed, failed to delete", msg_id)

    except Exception as e:
        logger.error("[QueueWorker] Error processing %s: %s", msg_id, e, exc_info=True)
        try:
            await delete_message("delayed_messages", msg_id)
        except Exception as delete_error:
            logger.error("[QueueWorker] Failed to delete %s: %s", msg_id, delete_error)
    finally:
        _in_flight_messages.discard(msg_id)


async def start_queue_worker() -> asyncio.Task:
    _shutdown_event.clear()
    _in_flight_messages.clear()

    task = asyncio.create_task(process_queue_worker())
    return task


async def graceful_shutdown_worker(timeout: float = GRACEFUL_SHUTDOWN_TIMEOUT) -> None:
    logger.info("[QueueWorker] Graceful shutdown started")
    _shutdown_event.set()

    if _in_flight_messages:
        logger.info("[QueueWorker] Waiting for %d in-flight messages", len(_in_flight_messages))
        try:
            await asyncio.wait_for(
                _wait_for_in_flight_completion(),
                timeout=timeout,
            )
            logger.info("[QueueWorker] All in-flight messages completed")
        except asyncio.TimeoutError:
            logger.warning(
                "[QueueWorker] Timeout waiting for in-flight messages. %d remaining",
                len(_in_flight_messages),
            )

    logger.info("[QueueWorker] Graceful shutdown complete")
