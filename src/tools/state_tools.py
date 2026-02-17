"""Инструменты для управления состоянием агента."""

from __future__ import annotations

import logging

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState

logger = logging.getLogger(__name__)


@tool
async def set_photo_requirement(
    require_photo: bool,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> Command:
    """Устанавливает требование наличия фотографий для товаров в state агента.
    
    Когда установлено require_photo=True, все инструменты поиска товаров
    будут возвращать только товары с фотографиями. Это влияет на:
    - vector_search: фильтрует результаты по наличию фото
    - execute_sql_query: фильтрует результаты по наличию фото
    - get_random_products: фильтрует результаты по наличию фото
    - get_product_by_title: проверяет наличие фото у найденного товара
    
    ИСПОЛЬЗУЙ КОГДА:
    - Клиент просит показать товары с фотографиями
    - Клиент хочет увидеть фото товаров
    - Клиент спрашивает "покажи товары с фото" или "есть ли фото"
    - Нужно отправить фотографии товаров через show_product_photos
    
    НЕ ИСПОЛЬЗУЙ ЕСЛИ:
    - Клиент не упоминает фотографии
    - Клиент просто ищет товары без требования фото
    - Нужно показать все товары независимо от наличия фото
    
    Args:
        require_photo: True если требуется наличие фото, False если не требуется
        runtime: ToolRuntime для доступа к context и state (автоматически инжектируется)
    
    Returns:
        Command для обновления state агента
    """
    try:
        status_text = "включено" if require_photo else "выключено"
        message_content = (
            f"Требование наличия фотографий {status_text}. "
            f"Все последующие поиски товаров будут {'возвращать только товары с фотографиями' if require_photo else 'возвращать все товары независимо от наличия фото'}."
        )
        
        tool_message = ToolMessage(
            content=message_content,
            tool_call_id="",  # LangGraph автоматически заполнит tool_call_id
        )
        
        logger.info(
            f"[set_photo_requirement] Установлено require_photo={require_photo} "
            f"для клиента {runtime.context.client_phone}"
        )
        
        return Command(
            update={
                "require_photo": require_photo,
                "messages": [tool_message],
            }
        )
        
    except Exception as e:
        logger.error(
            f"[set_photo_requirement] Ошибка при установке require_photo: {e}",
            exc_info=True,
        )
        error_message = ToolMessage(
            content=f"Ошибка при установке требования фото: {e}",
            tool_call_id="",
        )
        return Command(
            update={
                "require_photo": require_photo,
                "messages": [error_message],
            }
        )
