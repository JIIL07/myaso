"""Middleware для сохранения product_ids из artifacts инструментов поиска."""

from __future__ import annotations

import logging
from typing import Any, Awaitable, List

from langchain.agents.middleware import wrap_tool_call

from src.agents.tools.context_tools import save_product_ids_to_context
from src.agents.tools.context_vars import get_client_phone

logger = logging.getLogger(__name__)

# Инструменты поиска товаров, которые возвращают product_ids как artifacts
PRODUCT_SEARCH_TOOLS = {
    'vector_search',
    'execute_sql_query',
    'get_random_products',
    'find_similar_products',
    'get_product_by_title',
}


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
        # Проверяем, является ли результат ToolMessage с artifact
        if hasattr(result, 'artifact'):
            artifact = result.artifact
        # Если результат - кортеж (content, artifact) от инструмента с response_format="content_and_artifact"
        elif isinstance(result, tuple) and len(result) == 2:
            _, artifact = result
        else:
            # Не поддерживаемый формат
            return product_ids
        
        # Извлекаем product_ids из artifact
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
    """Middleware для сохранения product_ids из artifacts инструментов поиска.
    
    Перехватывает результат выполнения инструментов поиска товаров и сохраняет
    product_ids в контекст агента сразу после выполнения, чтобы они были доступны
    для других инструментов (например, show_product_photos) во время выполнения агента.
    
    Args:
        request: Запрос на выполнение инструмента
        handler: Асинхронный обработчик для выполнения инструмента
        
    Returns:
        Результат выполнения инструмента (без изменений)
    """
    # Выполняем инструмент
    # Согласно документации LangChain, handler всегда callable
    # и может возвращать как синхронный, так и асинхронный результат
    if callable(handler):
        result = handler(request)
        # Если результат асинхронный, ожидаем его
        if isinstance(result, Awaitable):
            result = await result
    else:
        # Fallback для случая, если handler не callable (не должно происходить)
        result = await handler if isinstance(handler, Awaitable) else handler
    
    # Получаем имя инструмента из request
    tool_name = None
    if hasattr(request, "tool_call"):
        if isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name")
        elif hasattr(request.tool_call, "name"):
            tool_name = request.tool_call.name
    
    # Если это инструмент поиска товаров, извлекаем и сохраняем product_ids
    if tool_name and tool_name in PRODUCT_SEARCH_TOOLS:
        try:
            product_ids = _extract_product_ids_from_result(result)
            
            if product_ids:
                # Получаем client_phone из контекста
                try:
                    client_phone = get_client_phone()
                    
                    # Сохраняем product_ids в контекст синхронно
                    # Это быстрое действие (запись в БД), не должно блокировать надолго
                    await _save_product_ids_async(client_phone, product_ids)
                    
                    logger.debug(
                        f"[product_ids_middleware] Найдено {len(product_ids)} product_ids "
                        f"в инструменте {tool_name}, сохранено в контекст для {client_phone}"
                    )
                except ValueError:
                    # client_phone не установлен в контексте - это нормально для некоторых случаев
                    logger.debug(
                        f"[product_ids_middleware] client_phone не установлен в контексте, "
                        f"пропускаем сохранение product_ids"
                    )
        except Exception as e:
            # Не прерываем выполнение при ошибке сохранения
            logger.warning(
                f"[product_ids_middleware] Ошибка при сохранении product_ids из {tool_name}: {e}",
                exc_info=True
            )
    
    return result


async def _save_product_ids_async(client_phone: str, product_ids: List[int]) -> None:
    """Асинхронно сохраняет product_ids в контекст.
    
    Args:
        client_phone: Номер телефона клиента
        product_ids: Список product_ids для сохранения
    """
    try:
        # Удаляем дубликаты, сохраняя порядок
        unique_ids = list(dict.fromkeys(product_ids))
        
        # Сохраняем в контекст
        await save_product_ids_to_context(client_phone, unique_ids)
        
        logger.info(
            f"[product_ids_middleware] Сохранено {len(unique_ids)} product_ids "
            f"в контекст для {client_phone}"
        )
    except Exception as e:
        logger.error(
            f"[product_ids_middleware] Ошибка сохранения product_ids для {client_phone}: {e}",
            exc_info=True
        )
