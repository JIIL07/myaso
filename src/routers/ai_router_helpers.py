"""Вспомогательные функции для ai_router."""

import logging
import os
from typing import Optional
from urllib.parse import urlparse

from src.config.messages_constants import (
    ERROR_MESSAGE_WHATSAPP_FAILED,
    PROMPT_TOPIC_WELCOME_MESSAGE,
    SYSTEM_VALUE_PRICELIST,
)
from src.services.whatsapp_service import send_image, send_message
from src.utils.async_mixin import remove_markdown_symbols
from src.utils.prompts import get_prompt, get_system_value

logger = logging.getLogger(__name__)


async def prepare_user_input_for_agent(
    user_input: str,
    is_init: bool = False,
    topic: Optional[str] = None
) -> str:
    """Подготавливает пользовательский ввод для агента.
    
    Инструкции загружаются динамически из БД и добавляются в системный промпт,
    поэтому здесь просто возвращаем user_input без изменений.
    
    Args:
        user_input: Исходный текст пользователя
        is_init: True если это инициализация беседы (не используется, оставлено для совместимости)
        topic: Тема беседы (не используется, оставлено для совместимости)
    
    Returns:
        Текст запроса пользователя без изменений
    """
    return user_input


async def send_pricelist_if_needed(client_phone: str) -> None:
    """Отправляет прайс-лист клиенту, если он настроен.
    
    Args:
        client_phone: Номер телефона клиента
    """
    try:
        pricelist_url = await get_system_value(SYSTEM_VALUE_PRICELIST)
        if not pricelist_url:
            logger.info(f"[send_pricelist_if_needed] Прайс-лист не настроен для {client_phone}")
            return
        
        parsed_url = urlparse(pricelist_url)
        file_path = parsed_url.path
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension:
            file_extension = file_extension.lstrip('.').lower()
        else:
            file_extension = "xlsx"
        
        send_file_success = await send_image(
            recipient=client_phone,
            file_url=pricelist_url,
            caption=SYSTEM_VALUE_PRICELIST,
            extension=file_extension,
        )
        
        if not send_file_success:
            logger.warning(
                f"[send_pricelist_if_needed] Не удалось отправить прайс-лист для {client_phone}"
            )
        else:
            logger.info(
                f"[send_pricelist_if_needed] Прайс-лист успешно отправлен для {client_phone}"
            )
    except Exception as pricelist_error:
        logger.error(
            f"[send_pricelist_if_needed] Ошибка при отправке прайс-листа для {client_phone}: {pricelist_error}",
            exc_info=True,
        )


async def send_agent_response(
    client_phone: str,
    response_text: str,
    endpoint_name: str
) -> bool:
    """Отправляет ответ агента клиенту через WhatsApp.
    
    Args:
        client_phone: Номер телефона клиента
        response_text: Текст ответа агента
        endpoint_name: Имя endpoint для логирования
    
    Returns:
        True если сообщение отправлено успешно, False иначе
    """
    try:
        await send_message(
            client_phone,
            remove_markdown_symbols(response_text),
        )
        logger.info(f"[{endpoint_name}] Ответ успешно отправлен для {client_phone}")
        return True
    except Exception as e:
        logger.error(
            f"[{endpoint_name}] Ошибка отправки в WhatsApp для {client_phone}: {e}",
            exc_info=True
        )
        return False


async def send_error_message(client_phone: str, endpoint_name: str) -> None:
    """Отправляет сообщение об ошибке клиенту.
    
    Args:
        client_phone: Номер телефона клиента
        endpoint_name: Имя endpoint для логирования
    """
    try:
        await send_message(client_phone, ERROR_MESSAGE_WHATSAPP_FAILED)
    except Exception as e:
        logger.error(
            f"[{endpoint_name}] Ошибка отправки сообщения об ошибке для {client_phone}: {e}"
        )

