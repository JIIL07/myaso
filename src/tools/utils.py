"""Утилиты для работы с инструментами агента."""
from typing import Optional

from langchain.tools import ToolRuntime

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState


def get_require_photo_from_runtime(
    runtime: Optional[ToolRuntime[ProductAgentContext, ProductAgentState]],
) -> bool:
    """Получает требование наличия фото из state агента.
    
    Args:
        runtime: ToolRuntime для доступа к state
        
    Returns:
        True если требуется наличие фото, False иначе
    """
    if not runtime:
        return False
    return runtime.state.get("require_photo", False)


def calculate_search_limit(limit: int, require_photo: bool, multiplier: int = 5) -> int:
    """Рассчитывает лимит для поиска с учетом требования фото.
    
    Если требуется фото, увеличивает лимит для последующей фильтрации.
    
    Args:
        limit: Исходный лимит
        require_photo: Требуется ли наличие фото
        multiplier: Множитель для увеличения лимита при требовании фото
        
    Returns:
        Рассчитанный лимит для поиска
    """
    return (limit * multiplier) if require_photo else limit
