"""Middleware для обработки ошибок инструментов."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


@wrap_tool_call
async def handle_tool_errors(request: Any, handler: Any) -> Any:
    """Асинхронно обрабатывает ошибки выполнения инструментов с понятными сообщениями.

    Args:
        request: Запрос на выполнение инструмента
        handler: Асинхронный обработчик для выполнения инструмента (может быть Callable или Awaitable)

    Returns:
        Результат выполнения инструмента или ToolMessage с ошибкой
    """
    try:
        # Проверяем, является ли handler корутиной (async функцией)
        if isinstance(handler, Awaitable):
            result = await handler
        elif callable(handler):
            # Проверяем, является ли handler async функцией
            result = handler(request)
            if isinstance(result, Awaitable):
                result = await result
        else:
            result = handler
        
        return result
    except Exception as e:
        # Извлекаем информацию об инструменте
        tool_name = "unknown"
        tool_call_id = ""
        
        if hasattr(request, "tool_call"):
            if isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get("name", "unknown")
                tool_call_id = request.tool_call.get("id", "")
            elif hasattr(request.tool_call, "name"):
                tool_name = request.tool_call.name
                tool_call_id = getattr(request.tool_call, "id", "")
        
        error_msg = str(e)

        logger.error(
            f"[handle_tool_errors] Ошибка выполнения инструмента {tool_name}: {error_msg}",
            exc_info=True
        )

        # Специальная обработка для show_product_photos
        if tool_name == "show_product_photos":
            return ToolMessage(
                content=(
                    "НЕ ОТПРАВЛЕНО: Произошла ошибка при отправке фотографий. "
                    "ВАЖНО: Несмотря на ошибку отправки фото, ты ДОЛЖЕН продолжить и "
                    "сформировать текстовое сообщение с информацией о товарах для клиента."
                ),
                tool_call_id=tool_call_id,
            )

        # Общая обработка для других инструментов
        return ToolMessage(
            content=f"Ошибка выполнения инструмента {tool_name}: {error_msg}. Попробуйте другой подход или уточните запрос.",
            tool_call_id=tool_call_id,
        )

