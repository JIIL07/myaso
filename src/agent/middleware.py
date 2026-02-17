"""Middleware для агентов - объединенный файл.

Содержит все middleware для обработки вызовов модели и инструментов:
- create_model_retry_middleware: retry вызовов модели с exponential backoff
- save_product_ids_middleware: сохранение product_ids из artifacts в state
- handle_tool_errors: обработка ошибок выполнения инструментов
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, List

from langchain.agents.middleware import (
    ModelRequest,
    ModelResponse,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langgraph.types import Command

logger = logging.getLogger(__name__)

# Константы для product_ids_middleware
PRODUCT_SEARCH_TOOLS = {
    'vector_search',
    'execute_sql_query',
    'get_random_products',
    'get_product_by_title',
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
    """Создает middleware для retry вызовов модели с exponential backoff.

    Args:
        max_retries: Максимальное количество повторных попыток (по умолчанию 2, итого 3 попытки)
        backoff_factor: Множитель для exponential backoff (по умолчанию 2.0)
        initial_delay: Начальная задержка в секундах перед первой попыткой (по умолчанию 1.0)
        max_delay: Максимальная задержка в секундах между попытками (по умолчанию 60.0)
        jitter: Добавлять ли случайную вариацию (±25%) к задержке (по умолчанию True)
        retry_on: Кортеж типов исключений для retry или функция для проверки (по умолчанию все исключения)
        on_failure: Поведение при исчерпании всех попыток: 'error' (выбросить исключение),
                   'continue' (вернуть AIMessage с ошибкой) или callable (по умолчанию 'error')

    Returns:
        Декорированная функция middleware для использования в create_agent
    """
    @wrap_model_call
    async def model_retry_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Middleware для автоматического retry вызовов модели с exponential backoff."""
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
                    logger.debug(
                        f"[model_retry_middleware] Исключение {type(e).__name__} не в списке для retry, "
                        f"пропускаем retry"
                    )
                    raise
                
                if attempt >= max_retries:
                    logger.warning(
                        f"[model_retry_middleware] Исчерпаны все попытки ({max_retries + 1}) "
                        f"для вызова модели: {e}"
                    )
                    
                    if on_failure == "error":
                        raise
                    elif on_failure == "continue":
                        from langchain_core.messages import AIMessage
                        return ModelResponse(
                            result=[
                                AIMessage(
                                    content=f"Ошибка вызова модели после {max_retries + 1} попыток: {e}"
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
                    delay = max(0, delay)  # Убеждаемся, что задержка не отрицательная
                
                logger.info(
                    f"[model_retry_middleware] Попытка {attempt + 1}/{max_retries + 1} не удалась: {e}. "
                    f"Повтор через {delay:.2f} секунд"
                )
                
                await asyncio.sleep(delay)
        
        if last_exception:
            raise last_exception
        raise RuntimeError("Неожиданная ошибка в model_retry_middleware")
    
    return model_retry_middleware


def _extract_product_ids_from_result(result: Any) -> List[int]:
    """Извлекает product_ids из результата инструмента.
    
    Поддерживает два формата:
    1. Кортеж (content, artifact) от инструмента с response_format="content_and_artifact"
    2. ToolMessage с атрибутом artifact (если LangChain автоматически создал ToolMessage)
    
    Args:
        result: Результат выполнения инструмента (может быть кортежом (content, artifact) или ToolMessage)
        
    Returns:
        Список product_ids
    """
    product_ids = []
    
    try:
        if hasattr(result, 'artifact'):
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
        logger.debug(
            f"[product_ids_middleware] Ошибка извлечения product_ids из результата: {e}"
        )
    
    return product_ids


@wrap_tool_call
async def save_product_ids_middleware(request: Any, handler: Any) -> Any:
    """Middleware для сохранения product_ids из artifacts инструментов поиска в state.
    
    Перехватывает результат выполнения инструментов поиска товаров и сохраняет
    product_ids в state агента через Command, чтобы они были доступны
    для других инструментов (например, show_product_photos) во время выполнения агента.
    
    Args:
        request: Запрос на выполнение инструмента
        handler: Асинхронный обработчик для выполнения инструмента
        
        Returns:
        Command для обновления state или исходный результат
    """
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
                current_product_ids = []
                if hasattr(request, "runtime") and request.runtime:
                    current_product_ids = request.runtime.state.get("product_ids", [])
                    client_phone = getattr(request.runtime.context, "client_phone", None)
                else:
                    client_phone = None
                
                all_ids = current_product_ids + product_ids
                unique_ids = list(dict.fromkeys(all_ids))
                
                logger.debug(
                    f"[product_ids_middleware] Добавлено {len(product_ids)} product_ids "
                    f"от инструмента {tool_name}, всего в state: {len(unique_ids)} "
                    f"(было: {len(current_product_ids)}, новых: {len(product_ids)})"
                )
                
                if isinstance(result, ToolMessage):
                    return Command(
                        update={
                            "product_ids": unique_ids,
                            "messages": [result]
                        }
                    )
                elif isinstance(result, tuple) and len(result) == 2:
                    content, artifact = result
                    tool_message = ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id or "",
                        artifact=artifact
                    )
                    return Command(
                        update={
                            "product_ids": unique_ids,
                            "messages": [tool_message]
                        }
                    )
        except Exception as e:
            logger.warning(
                f"[product_ids_middleware] Ошибка при сохранении product_ids из {tool_name}: {e}",
                exc_info=True
            )
    
    return result


def _get_error_message(error: Exception, tool_name: str) -> str:
    """Генерирует понятное сообщение об ошибке в зависимости от типа исключения.

    Args:
        error: Исключение, которое произошло
        tool_name: Имя инструмента, в котором произошла ошибка

    Returns:
        Понятное сообщение об ошибке для агента
    """
    if isinstance(error, ConnectionError):
        return (
            f"Сервис временно недоступен, но безопасно повторить попытку. "
            f"Попробуйте снова с теми же параметрами для инструмента {tool_name}."
        )
    elif isinstance(error, TimeoutError) or isinstance(error, asyncio.TimeoutError):
        return (
            f"Операция заняла слишком много времени для инструмента {tool_name}. "
            f"Попробуйте упростить запрос или разбить его на более мелкие части."
        )
    elif isinstance(error, ValueError):
        return (
            f"Ошибка валидации данных для инструмента {tool_name}: {str(error)}. "
            f"Проверьте входные данные и попробуйте снова."
        )
    elif isinstance(error, KeyError):
        return (
            f"Отсутствует необходимый параметр для инструмента {tool_name}: {str(error)}. "
            f"Проверьте, что все необходимые данные переданы."
        )
    elif isinstance(error, PermissionError):
        return (
            f"Нет доступа для выполнения операции инструмента {tool_name}. "
            f"Проверьте права доступа."
        )
    else:
        return (
            f"Ошибка выполнения инструмента {tool_name}: {str(error)}. "
            f"Попробуйте другой подход или уточните запрос."
        )


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
        
        error_msg = str(e)

        logger.error(
            f"[handle_tool_errors] Ошибка выполнения инструмента {tool_name}: {error_msg}",
            exc_info=True
        )

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
