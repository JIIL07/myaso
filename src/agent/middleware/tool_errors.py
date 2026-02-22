from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage

logger = logging.getLogger(__name__)


def _get_error_message(error: Exception, tool_name: str) -> str:
    if isinstance(error, ConnectionError):
        return (
            "Сервис временно недоступен, но безопасно повторить попытку. "
            "Попробуйте снова с теми же параметрами для инструмента %s." % tool_name
        )
    elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return (
            "Операция заняла слишком много времени для инструмента %s. "
            "Попробуйте упростить запрос или разбить его на более мелкие части." % tool_name
        )
    elif isinstance(error, ValueError):
        return (
            "Ошибка валидации данных для инструмента %s: %s. "
            "Проверьте входные данные и попробуйте снова." % (tool_name, error)
        )
    elif isinstance(error, KeyError):
        return (
            "Отсутствует необходимый параметр для инструмента %s: %s. "
            "Проверьте, что все необходимые данные переданы." % (tool_name, error)
        )
    elif isinstance(error, PermissionError):
        return (
            "Нет доступа для выполнения операции инструмента %s. "
            "Проверьте права доступа." % tool_name
        )
    else:
        return (
            "Ошибка выполнения инструмента %s: %s. "
            "Попробуйте другой подход или уточните запрос." % (tool_name, error)
        )


@wrap_tool_call
async def handle_tool_errors(request: Any, handler: Any) -> Any:
    try:
        if callable(handler):
            result = handler(request)
            if isinstance(result, Awaitable):
                result = await result
        else:
            result = await handler if isinstance(handler, Awaitable) else handler

        return result
    except Exception as e:
        tool_name = "unknown"
        tool_call_id = ""

        if hasattr(request, "tool_call"):
            if isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get("name", "unknown")
                tool_call_id = request.tool_call.get("id", "")
            elif hasattr(request.tool_call, "name"):
                tool_name = request.tool_call.name
                tool_call_id = getattr(request.tool_call, "id", "")

        logger.error("[ToolErrors] Error %s: %s", tool_name, e, exc_info=True)

        error_content = _get_error_message(e, tool_name)

        if tool_name == "show_product_photos":
            return ToolMessage(
                content=(
                    "НЕ ОТПРАВЛЕНО: Произошла ошибка при отправке фотографий. "
                    "ВАЖНО: Несмотря на ошибку отправки фото, ты ДОЛЖЕН продолжить и "
                    "сформировать текстовое сообщение с информацией о товарах для клиента."
                ),
                tool_call_id=tool_call_id,
            )

        return ToolMessage(
            content=error_content,
            tool_call_id=tool_call_id,
        )

