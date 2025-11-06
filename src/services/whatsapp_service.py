"""Сервис для работы с WhatsApp API."""

import logging
import httpx
from typing import Optional
from src.config.settings import settings
from src.config.constants import HTTP_TIMEOUT_SECONDS

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


async def send_image(recipient: str, image_url: str, caption: Optional[str] = None) -> bool:
    """Отправляет изображение через WhatsApp API.

    Args:
        recipient: Номер телефона получателя
        image_url: URL изображения
        caption: Подпись к изображению (опционально)

    Returns:
        True если изображение отправлено успешно, False иначе
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                settings.whatsapp.send_image_url,
                json={
                    "recipient": recipient,
                    "image_url": image_url,
                    "caption": caption or "",
                },
            )
            response.raise_for_status()
            return True
    except Exception as e:
        logger.error(f"Ошибка отправки изображения в WhatsApp для {recipient}: {e}")
        return False

