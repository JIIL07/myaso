"""Утилита для выполнения HTTP запросов с обработкой ошибок."""

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


async def send_http_post(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
    service_name: str = "http",
    recipient: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> bool:
    """Выполняет HTTP POST запрос с унифицированной обработкой ошибок.

    Args:
        url: URL для запроса
        payload: Данные для отправки (JSON)
        timeout: Таймаут запроса в секундах
        service_name: Название сервиса для логирования (например, "whatsapp", "telegram")
        recipient: Получатель (для логирования)
        additional_context: Дополнительный контекст для логирования (например, file_url)

    Returns:
        True если запрос успешен, False в случае ошибки
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url=url, json=payload)
            response.raise_for_status()
            return True
    except httpx.HTTPStatusError as e:
        context_parts = [f"HTTP ошибка для {recipient}"] if recipient else ["HTTP ошибка"]
        if additional_context:
            context_parts.append(f", {additional_context}")
        context_parts.append(f": статус {e.response.status_code}")
        
        logger.error(
            f"[{service_name}] {''.join(context_parts)}"
        )
        return False
    except httpx.TimeoutException:
        context = f"для {recipient}" if recipient else ""
        if additional_context:
            context += f", {additional_context}"
        
        logger.error(
            f"[{service_name}] Таймаут при отправке {context}"
        )
        return False
    except Exception as e:
        context = f"для {recipient}" if recipient else ""
        if additional_context:
            context += f", {additional_context}"
        
        logger.error(
            f"[{service_name}] Ошибка {context}: {e}",
            exc_info=True,
        )
        return False
