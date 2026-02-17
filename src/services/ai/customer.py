"""Сервис для работы с клиентами."""
import asyncio
import logging
from typing import Optional

from src.queries.clients_queries import get_client_by_phone, get_client_send_message
from src.queries.history_queries import get_conversation_history_count
from src.utils.logger.masking import mask_phone
from .exceptions import ClientValidationError

logger = logging.getLogger(__name__)


class CustomerService:
    """Сервис для валидации и проверки клиентов."""

    @staticmethod
    async def validate_client_exists(client_phone: str) -> None:
        """Проверяет, существует ли клиент в БД.

        Args:
            client_phone: Номер телефона клиента

        Raises:
            ClientValidationError: Если клиент не найден
        """
        client = await get_client_by_phone(client_phone)
        if client is None:
            logger.info(f"Клиент не найден в БД: {client_phone}")
            raise ClientValidationError("Client not found in database")

    @staticmethod
    async def validate_message_sending_enabled(client_phone: str) -> None:
        """Проверяет, разрешена ли отправка сообщений клиенту.

        Args:
            client_phone: Номер телефона клиента

        Raises:
            ClientValidationError: Если отправка сообщений отключена
        """
        send_message_enabled = await get_client_send_message(client_phone)
        if not send_message_enabled:
            logger.info(f"Отправка сообщений отключена для: {client_phone}")
            raise ClientValidationError("Message sending disabled")

    @staticmethod
    async def validate_conversation_initialized(client_phone: str) -> None:
        """Проверяет, инициализирован ли разговор.

        Args:
            client_phone: Номер телефона клиента

        Raises:
            ClientValidationError: Если разговор не инициализирован
        """
        history_count = await get_conversation_history_count(client_phone)
        if history_count == 0:
            logger.info(f"Разговор не инициализирован для: {client_phone}")
            raise ClientValidationError("Conversation not initialized")

    @classmethod
    async def validate_client_for_conversation(cls, client_phone: str) -> None:
        """Полная валидация клиента для обработки сообщений.

        Выполняет параллельные запросы к БД для оптимизации производительности.

        Args:
            client_phone: Номер телефона клиента

        Raises:
            ClientValidationError: Если валидация не прошла
        """
        # Параллельное выполнение запросов для оптимизации
        client, history_count = await asyncio.gather(
            get_client_by_phone(client_phone),
            get_conversation_history_count(client_phone),
        )
        
        # Валидация существования клиента
        if client is None:
            logger.info(f"Клиент не найден в БД: {mask_phone(client_phone)}")
            raise ClientValidationError("Client not found in database")
        
        # Валидация разрешения отправки сообщений
        send_message_enabled = client.get("send_message", True)
        if not send_message_enabled:
            logger.info(f"Отправка сообщений отключена для: {mask_phone(client_phone)}")
            raise ClientValidationError("Message sending disabled")
        
        # Валидация инициализации разговора
        if history_count == 0:
            logger.info(f"Разговор не инициализирован для: {mask_phone(client_phone)}")
            raise ClientValidationError("Conversation not initialized")
