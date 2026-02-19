from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

PRODUCT_SEARCH_TOOLS = {
    "vector_search",
    "execute_sql_query",
    "get_random_products",
    "get_product_by_title",
}


def create_model_retry_middleware(
    max_retries: int = 2,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] | Callable[[Exception], bool] | None = None,
    on_failure: str | Callable[[Exception], ModelResponse] = "error",
) -> Callable:
    @wrap_model_call
    async def model_retry_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | Awaitable[ModelResponse]],
    ) -> ModelResponse:
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = handler(request)
                if isinstance(result, Awaitable):
                    result = await result
                return result
            except Exception as e:
                last_exception = e

                should_retry = True
                if retry_on is not None:
                    if callable(retry_on):
                        should_retry = retry_on(e)
                    else:
                        should_retry = isinstance(e, retry_on)

                if not should_retry:
                    logger.debug("[ModelRetry] %s not in retry list", type(e).__name__)
                    raise

                if attempt >= max_retries:
                    logger.warning(
                        "[ModelRetry] All attempts exhausted (%d): %s",
                        max_retries + 1, e,
                    )

                    if on_failure == "error":
                        raise
                    elif on_failure == "continue":
                        from langchain_core.messages import AIMessage
                        return ModelResponse(
                            result=[
                                AIMessage(
                                    content="Model call error after %d attempts: %s"
                                    % (max_retries + 1, e)
                                )
                            ]
                        )
                    elif callable(on_failure):
                        return on_failure(e)
                    else:
                        raise

                delay = initial_delay * (backoff_factor ** attempt)
                delay = min(delay, max_delay)

                if jitter:
                    jitter_amount = delay * 0.25
                    delay = delay + random.uniform(-jitter_amount, jitter_amount)
                    delay = max(0, delay)

                logger.info(
                    "[ModelRetry] Attempt %d/%d failed: %s. Retrying in %.2fs",
                    attempt + 1, max_retries + 1, e, delay,
                )

                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("[ModelRetry] Unexpected error")

    return model_retry_middleware


def _extract_product_ids_from_result(result: Any) -> list[int]:
    product_ids: list[int] = []

    try:
        if hasattr(result, "artifact"):
            artifact = result.artifact
        elif isinstance(result, tuple) and len(result) == 2:
            _, artifact = result
        else:
            return product_ids

        if isinstance(artifact, list):
            for item in artifact:
                if isinstance(item, (int, str)):
                    product_id = int(item)
                    if product_id > 0:
                        product_ids.append(product_id)
        elif isinstance(artifact, (int, str)):
            product_id = int(artifact)
            if product_id > 0:
                product_ids.append(product_id)
    except (ValueError, TypeError) as e:
        logger.debug("[ProductIds] Error extracting product_ids: %s", e)

    return product_ids


@wrap_tool_call
async def save_product_ids_middleware(request: Any, handler: Any) -> Any:
    if callable(handler):
        result = handler(request)
        if isinstance(result, Awaitable):
            result = await result
    else:
        result = await handler if isinstance(handler, Awaitable) else handler

    tool_name = None
    tool_call_id = None
    if hasattr(request, "tool_call"):
        if isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name")
            tool_call_id = request.tool_call.get("id")
        elif hasattr(request.tool_call, "name"):
            tool_name = request.tool_call.name
            tool_call_id = getattr(request.tool_call, "id", None)

    if tool_name and tool_name in PRODUCT_SEARCH_TOOLS:
        try:
            product_ids = _extract_product_ids_from_result(result)

            if product_ids:
                current_product_ids: list[int] = []
                if hasattr(request, "runtime") and request.runtime:
                    current_product_ids = request.runtime.state.get("product_ids", [])

                all_ids = current_product_ids + product_ids
                unique_ids = list(dict.fromkeys(all_ids))

                logger.debug(
                    "[ProductIds] +%d from %s, total: %d",
                    len(product_ids), tool_name, len(unique_ids),
                )

                if isinstance(result, ToolMessage):
                    return Command(
                        update={
                            "product_ids": unique_ids,
                            "messages": [result],
                        }
                    )
                elif isinstance(result, tuple) and len(result) == 2:
                    content, artifact = result
                    tool_message = ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id or "",
                        artifact=artifact,
                    )
                    return Command(
                        update={
                            "product_ids": unique_ids,
                            "messages": [tool_message],
                        }
                    )
        except Exception as e:
            logger.warning("[ProductIds] Error saving from %s: %s", tool_name, e, exc_info=True)

    return result


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
