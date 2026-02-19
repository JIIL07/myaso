"""Сервис для отправки сообщений через Telegram."""

import asyncio
from typing import Optional

from src.services.messaging.base import BaseMessagingService
from src.constants import ERROR_MESSAGE_TELEGRAM_FAILED, HTTP_TIMEOUT_SECONDS

_telegram_service: Optional[BaseMessagingService] = None
_lock = asyncio.Lock()


def _get_telegram_service() -> BaseMessagingService:
    """Получает или создает singleton экземпляр сервиса.
    
    Использует asyncio.Lock для thread-safety при параллельных запросах.
    """
    global _telegram_service
    
    if _telegram_service is not None:
        return _telegram_service
    
    # Для sync функции используем синхронную блокировку через threading
    import threading
    sync_lock = threading.Lock()
    
    with sync_lock:
        # Double-check после получения блокировки
        if _telegram_service is None:
            from src.config.settings import settings
            _telegram_service = BaseMessagingService(
                send_message_url=settings.telegram.send_message_url,
                send_file_url=settings.telegram.send_file_url,
                timeout=HTTP_TIMEOUT_SECONDS,
                error_message=ERROR_MESSAGE_TELEGRAM_FAILED,
                service_name="telegram",
                recipient_name="ID получателя",
            )
    
    return _telegram_service


class TelegramMessagingService:
    """Сервис для отправки различных типов сообщений через Telegram."""

    @staticmethod
    async def send_text_message(
        client_phone: str,
        message_text: str,
        remove_markdown: bool = True,
        context: str = "",
    ) -> bool:
        """Отправляет текстовое сообщение клиенту.

        Args:
            client_phone: ID чата клиента в Telegram (или номер телефона для совместимости)
            message_text: Текст сообщения
            remove_markdown: Удалить markdown символы (по умолчанию True)
            context: Контекст для логирования (имя endpoint)

        Returns:
            True если сообщение отправлено успешно, False иначе
        """
        return await _get_telegram_service().send_text_message(
            client_id=client_phone,
            message_text=message_text,
            remove_markdown=remove_markdown,
            context=context,
        )

    @staticmethod
    async def send_error_message(client_phone: str, context: str = "") -> None:
        """Отправляет стандартное сообщение об ошибке клиенту.

        Args:
            client_phone: ID чата клиента в Telegram (или номер телефона для совместимости)
            context: Контекст для логирования
        """
        await _get_telegram_service().send_error_message(client_id=client_phone, context=context)

    @staticmethod
    async def send_pricelist(client_phone: str, context: str = "") -> bool:
        """Отправляет прайс-лист клиенту, если он настроен в системе.

        Args:
            client_phone: ID чата клиента в Telegram (или номер телефона для совместимости)
            context: Контекст для логирования

        Returns:
            True если прайс-лист отправлен, False иначе
        """
        return await _get_telegram_service().send_pricelist(client_id=client_phone, context=context)


async def send_message(recipient: str, message: str) -> bool:
    """Отправляет текстовое сообщение через Telegram API.

    Args:
        recipient: ID чата получателя
        message: Текст сообщения

    Returns:
        True если сообщение успешно отправлено, False в случае ошибки
    """
    return await _get_telegram_service().send_message(recipient, message)


async def send_image(
    recipient: str, file_url: str, caption: str = "", extension: str = "png"
) -> bool:
    """Отправляет файл через Telegram API.

    Args:
        recipient: ID чата получателя
        file_url: URL файла для отправки
        caption: Подпись к файлу
        extension: Расширение файла (по умолчанию "png")

    Returns:
        True если файл успешно отправлен, False в случае ошибки
    """
    return await _get_telegram_service().send_image(recipient, file_url, caption, extension)
