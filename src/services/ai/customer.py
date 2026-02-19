import asyncio
import logging

from src.queries.clients_queries import get_client_by_phone, get_client_send_message
from src.queries.history_queries import get_conversation_history_count
from src.utils.logger.masking import mask_phone
from .exceptions import ClientValidationError

logger = logging.getLogger(__name__)


class CustomerService:
    @staticmethod
    async def validate_client_exists(client_phone: str) -> None:
        client = await get_client_by_phone(client_phone)
        if client is None:
            logger.info("[Customer] Client not found: %s", client_phone)
            raise ClientValidationError("Client not found in database")

    @staticmethod
    async def validate_message_sending_enabled(client_phone: str) -> None:
        send_message_enabled = await get_client_send_message(client_phone)
        if not send_message_enabled:
            raise ClientValidationError("Message sending disabled")

    @staticmethod
    async def validate_conversation_initialized(client_phone: str) -> None:
        history_count = await get_conversation_history_count(client_phone)
        if history_count == 0:
            raise ClientValidationError("Conversation not initialized")

    @classmethod
    async def validate_client_for_conversation(cls, client_phone: str) -> None:
        client, history_count = await asyncio.gather(
            get_client_by_phone(client_phone),
            get_conversation_history_count(client_phone),
        )
        if client is None:
            logger.info("[Customer] Client not found: %s", mask_phone(client_phone))
            raise ClientValidationError("Client not found in database")
        if not client.get("send_message", True):
            raise ClientValidationError("Message sending disabled")
        if history_count == 0:
            raise ClientValidationError("Conversation not initialized")
