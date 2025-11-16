"""Сервис для работы с WhatsApp API."""

import logging
from typing import Optional

import httpx

from src.config.constants import HTTP_TIMEOUT_SECONDS
from src.config.settings import settings

logger = logging.getLogger(__name__)


async def send_message(recipient: str, message: str) -> bool:
    """Отправляет текстовое сообщение через WhatsApp API.

    Args:
        recipient: Номер телефона получателя
        message: Текст сообщения

    Returns:
        True если сообщение отправлено успешно, False иначе
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                settings.whatsapp.send_message_url,
                json={"recipient": recipient, "message": message},
            )
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения в WhatsApp для {recipient}: {e}")
        return False


async def send_image(recipient: str, file_url: str, caption: Optional[str] = None, extension: str = "png") -> bool:
    """Отправляет файл через WhatsApp API.

    Args:
        recipient: Номер телефона получателя
        file_url: URL файла
        caption: Подпись к файлу (опционально)
        extension: Тип файла (по умолчанию "png")

    Returns:
        True если файл отправлен успешно, False иначе
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                settings.whatsapp.send_file_url,
                json={
                    "recipient": recipient,
                    "file_url": file_url,
                    "caption": caption or "",
                    "extension": extension,
                },
            )
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Ошибка отправки файла в WhatsApp для {recipient}: {e}")
        return False

