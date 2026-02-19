"""Tool: set_photo_requirement — toggle photo-only mode for product searches."""

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
    """Включает/выключает фильтр «только товары с фото» для всех поисковых инструментов.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Клиент хочет видеть товары с фотографиями
    - Нужно отправить фото через show_product_photos

    НЕ ИСПОЛЬЗОВАТЬ:
    - Клиент не упоминает фотографии
    - Нужно показать все товары независимо от наличия фото
    """
    try:
        status_text = "включено" if require_photo else "выключено"
        message_content = (
            "Требование наличия фотографий %s. "
            "Все последующие поиски товаров будут %s."
            % (
                status_text,
                "возвращать только товары с фотографиями"
                if require_photo
                else "возвращать все товары независимо от наличия фото",
            )
        )

        logger.info(
            "[set_photo_requirement] require_photo=%s for %s",
            require_photo,
            runtime.context.client_phone,
        )

        return Command(
            update={
                "require_photo": require_photo,
                "messages": [ToolMessage(content=message_content, tool_call_id="")],
            }
        )
    except Exception as e:
        logger.error("[set_photo_requirement] Error: %s", e, exc_info=True)
        return Command(
            update={
                "require_photo": require_photo,
                "messages": [
                    ToolMessage(
                        content="Ошибка при установке требования фото: %s" % e,
                        tool_call_id="",
                    )
                ],
            }
        )
