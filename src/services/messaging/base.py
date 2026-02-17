"""Базовый класс для messaging сервисов (WhatsApp, Telegram)."""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

from src.services.ai.constants import SYSTEM_VALUE_PRICELIST
from src.services.ai.prompt import get_system_value
from src.utils import remove_markdown_symbols
from src.utils.http.http_client import send_http_post
from src.utils.validators.string_validators import validate_not_empty

logger = logging.getLogger(__name__)


class BaseMessagingService:
    """Базовый класс для сервисов отправки сообщений."""

    def __init__(
        self,
        send_message_url: str,
        send_file_url: str,
        timeout: float,
        error_message: str,
        service_name: str,
        recipient_name: str = "получатель",
    ):
        """Инициализирует базовый messaging сервис.

        Args:
            send_message_url: URL для отправки текстовых сообщений
            send_file_url: URL для отправки файлов
            timeout: Таймаут HTTP запросов в секундах
            error_message: Сообщение об ошибке по умолчанию
            service_name: Название сервиса для логирования (например, "whatsapp", "telegram")
            recipient_name: Название получателя для логирования (например, "номер телефона", "ID получателя")
        """
        self.send_message_url = send_message_url
        self.send_file_url = send_file_url
        self.timeout = timeout
        self.error_message = error_message
        self.service_name = service_name
        self.recipient_name = recipient_name

    async def send_message(self, recipient: str, message: str) -> bool:
        """Отправляет текстовое сообщение.

        Args:
            recipient: Получатель (номер телефона или ID чата)
            message: Текст сообщения

        Returns:
            True если сообщение успешно отправлено, False в случае ошибки
        """
        if not validate_not_empty(recipient, self.recipient_name, "send_message"):
            return False

        if not validate_not_empty(message, "сообщение", "send_message"):
            return False

        return await send_http_post(
            url=self.send_message_url,
            payload={"recipient": recipient, "message": message},
            timeout=self.timeout,
            service_name=self.service_name,
            recipient=recipient,
        )

    async def send_image(
        self,
        recipient: str,
        file_url: str,
        caption: str = "",
        extension: str = "png",
    ) -> bool:
        """Отправляет файл (изображение).

        Args:
            recipient: Получатель (номер телефона или ID чата)
            file_url: URL файла для отправки
            caption: Подпись к файлу
            extension: Расширение файла (по умолчанию "png")

        Returns:
            True если файл успешно отправлен, False в случае ошибки
        """
        if not validate_not_empty(recipient, self.recipient_name, "send_image"):
            return False

        if not validate_not_empty(file_url, "URL файла", "send_image"):
            return False

        return await send_http_post(
            url=self.send_file_url,
            payload={
                "recipient": recipient,
                "file_url": file_url,
                "caption": caption,
                "extension": extension,
            },
            timeout=self.timeout,
            service_name=self.service_name,
            recipient=recipient,
            additional_context=f"файл: {file_url}",
        )

    async def send_text_message(
        self,
        client_id: str,
        message_text: str,
        remove_markdown: bool = True,
        context: str = "",
    ) -> bool:
        """Отправляет текстовое сообщение клиенту.

        Args:
            client_id: ID клиента (номер телефона или ID чата)
            message_text: Текст сообщения
            remove_markdown: Удалить markdown символы (по умолчанию True)
            context: Контекст для логирования (имя endpoint)

        Returns:
            True если сообщение отправлено успешно, False иначе
        """
        try:
            text_to_send = (
                remove_markdown_symbols(message_text) if remove_markdown else message_text
            )

            success = await self.send_message(client_id, text_to_send)

            if not success:
                logger.warning(
                    f"[{context}] Не удалось отправить сообщение для {client_id}"
                )

            return success

        except Exception as e:
            logger.error(
                f"[{context}] Ошибка отправки сообщения для {client_id}: {e}",
                exc_info=True,
            )
            return False

    async def send_error_message(self, client_id: str, context: str = "") -> None:
        """Отправляет стандартное сообщение об ошибке клиенту.

        Args:
            client_id: ID клиента (номер телефона или ID чата)
            context: Контекст для логирования
        """
        try:
            await self.send_message(client_id, self.error_message)
        except Exception as e:
            logger.error(
                f"[{context}] Не удалось отправить сообщение об ошибке для {client_id}: {e}",
                exc_info=True,
            )

    async def send_pricelist(self, client_id: str, context: str = "") -> bool:
        """Отправляет прайс-лист клиенту, если он настроен в системе.

        Args:
            client_id: ID клиента (номер телефона или ID чата)
            context: Контекст для логирования

        Returns:
            True если прайс-лист отправлен, False иначе
        """
        try:
            pricelist_url = await get_system_value(SYSTEM_VALUE_PRICELIST)

            if not pricelist_url:
                logger.info(
                    f"[{context}] Прайс-лист не настроен в системе для {client_id}"
                )
                return False

            logger.info(
                f"[{context}] Найден прайс-лист для {client_id}: {pricelist_url}"
            )

            parsed_url = urlparse(pricelist_url)
            file_path = parsed_url.path
            _, file_extension = os.path.splitext(file_path)
            file_extension = (
                file_extension.lstrip(".").lower() if file_extension else "xlsx"
            )

            success = await self.send_image(
                recipient=client_id,
                file_url=pricelist_url,
                caption="Прайс-лист",
                extension=file_extension,
            )

            if not success:
                logger.warning(
                    f"[{context}] Не удалось отправить прайс-лист для {client_id}"
                )

            return success

        except Exception as e:
            logger.error(
                f"[{context}] Ошибка при отправке прайс-листа для {client_id}: {e}",
                exc_info=True,
            )
            return False
