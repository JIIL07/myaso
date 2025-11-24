"""Инструменты для управления контекстом агента."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.config.constants import CONTEXT_CACHE_TTL_SECONDS
from src.database.queries.context_queries import (
    get_agent_context_from_db,
    save_agent_context_to_db,
)
from src.agents.tools.context_vars import get_client_phone

logger = logging.getLogger(__name__)

# In-memory кэш для контекста
_context_cache: Dict[str, tuple[Dict[str, Any], float]] = {}


async def get_agent_context_async(client_phone: str) -> Dict[str, Any]:
    """Асинхронно получает контекст агента для клиента (с кэшированием).

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Словарь с контекстом агента
    """
    current_time = time.time()
    
    # Проверяем кэш
    if client_phone in _context_cache:
        cached_context, cache_time = _context_cache[client_phone]
        if current_time - cache_time < CONTEXT_CACHE_TTL_SECONDS:
            return cached_context
        # Кэш устарел, удаляем
        del _context_cache[client_phone]
    
    # Загружаем из БД
    try:
        context = await get_agent_context_from_db(client_phone)
        # Если контекст пустой, создаем дефолтный
        if not context:
            context = {"require_photo": False}
            await save_agent_context_to_db(client_phone, context)
        # Сохраняем в кэш
        _context_cache[client_phone] = (context, current_time)
        return context
    except Exception as e:
        logger.error(f"Ошибка при загрузке контекста для {client_phone}: {e}", exc_info=True)
        # Возвращаем дефолтный контекст при ошибке
        return {"require_photo": False}


def get_agent_context(client_phone: str) -> Dict[str, Any]:
    """Получает контекст агента для клиента (синхронная обертка).

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Словарь с контекстом агента
    """
    # Используем asyncio для синхронного вызова асинхронной функции
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если событийный цикл уже запущен, создаем новый
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(get_agent_context_async(client_phone))
                )
                return future.result()
        else:
            return loop.run_until_complete(get_agent_context_async(client_phone))
    except RuntimeError:
        # Если нет событийного цикла, создаем новый
        return asyncio.run(get_agent_context_async(client_phone))


async def _save_agent_context_async(client_phone: str, context_data: Dict[str, Any]) -> None:
    """Асинхронно сохраняет контекст агента.

    Args:
        client_phone: Номер телефона клиента
        context_data: Данные контекста для сохранения
    """
    try:
        await save_agent_context_to_db(client_phone, context_data)
        # Обновляем кэш
        _context_cache[client_phone] = (context_data, time.time())
    except Exception as e:
        logger.error(f"Ошибка при сохранении контекста для {client_phone}: {e}", exc_info=True)


@tool
async def set_photo_requirement(require: bool) -> str:
    """Устанавливает требование наличия фотографий для поиска товаров.

    Используй этот инструмент когда нужно указать, что поиск должен возвращать
    только товары с фотографиями.

    Args:
        require: True если требуются только товары с фото, False если фото не обязательны

    Returns:
        Подтверждение установки требования
    """
    client_phone = get_client_phone()
    context = get_agent_context(client_phone)
    context["require_photo"] = require
    await _save_agent_context_async(client_phone, context)
    if require:
        return "Требование установлено: возвращать только товары с фотографиями"
    else:
        return "Требование снято: возвращать все товары независимо от наличия фото"


@tool
def get_conversation_context() -> str:
    """Получает текущий контекст разговора.

    Returns:
        Информация о текущем контексте
    """
    client_phone = get_client_phone()
    context = get_agent_context(client_phone)
    photo_req = "требуются" if context["require_photo"] else "не требуются"
    return f"Фото: {photo_req}"


def get_require_photo(client_phone: Optional[str] = None) -> bool:
    """Получает флаг require_photo для клиента.

    Args:
        client_phone: Номер телефона клиента (опционально, если None - берется из контекста)

    Returns:
        True если требуются только товары с фото, False иначе
    """
    if client_phone is None:
        client_phone = get_client_phone()
    context = get_agent_context(client_phone)
    return context.get("require_photo", False)


async def save_product_ids_to_context(client_phone: str, product_ids: List[int]) -> None:
    """Сохраняет список ID товаров в контекст агента для последующей отправки фотографий.
    
    Args:
        client_phone: Номер телефона клиента
        product_ids: Список ID товаров
    """
    try:
        context = await get_agent_context_async(client_phone)
        context["product_ids_for_photos"] = product_ids
        await _save_agent_context_async(client_phone, context)
        logger.info(f"[save_product_ids_to_context] Сохранено {len(product_ids)} ID товаров для {client_phone}")
    except Exception as e:
        logger.error(f"[save_product_ids_to_context] Ошибка сохранения product_ids: {e}", exc_info=True)


def get_product_ids_from_context(client_phone: Optional[str] = None) -> List[int]:
    """Получает список ID товаров из контекста агента.
    
    Args:
        client_phone: Номер телефона клиента (опционально, если None - берется из контекста)
        
    Returns:
        Список ID товаров
    """
    if client_phone is None:
        client_phone = get_client_phone()
    context = get_agent_context(client_phone)
    return context.get("product_ids_for_photos", [])

