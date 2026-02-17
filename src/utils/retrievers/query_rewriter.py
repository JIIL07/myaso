"""Модуль для переформулировки запросов на основе истории разговора."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.utils.retrievers.constants import ENABLE_QUERY_REWRITING, MAX_HISTORY_FOR_REWRITING

logger = logging.getLogger(__name__)


async def rewrite_query_with_context(
    query: str,
    chat_history: List[BaseMessage],
    client_context: Dict[str, Any],
) -> str:
    """Переформулирует запрос с учетом истории разговора и контекста клиента.

    Args:
        query: Исходный запрос пользователя
        chat_history: История сообщений
        client_context: Контекст клиента (предпочтения, настройки)

    Returns:
        Переформулированный запрос
    """
    if not ENABLE_QUERY_REWRITING:
        return query

    try:
        max_history = MAX_HISTORY_FOR_REWRITING
        
        recent_messages = chat_history[-max_history:] if chat_history else []
        
        history_context = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                history_context.append(f"Пользователь: {msg.content}")
            else:
                history_context.append(f"Ассистент: {msg.content[:200]}")
        
        history_text = "\n".join(history_context) if history_context else "История пуста"
        
        rewriting_prompt = f"""Переформулируй запрос пользователя с учетом контекста разговора.

История разговора:
{history_text}

Контекст клиента:
- Требуются фото: {client_context.get('require_photo', False)}

Исходный запрос: {query}

Переформулируй запрос, добавив релевантную информацию из истории, если это поможет найти более подходящие товары.
Если история не содержит релевантной информации, верни исходный запрос без изменений.

Переформулированный запрос:"""

        llm = ChatOpenAI(
            model=settings.openrouter.model_id,
            openai_api_key=settings.openrouter.openrouter_api_key,
            openai_api_base=settings.openrouter.base_url,
            temperature=0.3,
        )

        result = await llm.ainvoke(rewriting_prompt)
        rewritten_query = result.content.strip() if hasattr(result, 'content') else str(result).strip()
        
        logger.info(f"[query_rewriter] Исходный запрос: '{query}' -> Переформулированный: '{rewritten_query}'")
        return rewritten_query

    except Exception as e:
        logger.error(f"[query_rewriter] Ошибка при переформулировке запроса: {e}", exc_info=True)
        return query


