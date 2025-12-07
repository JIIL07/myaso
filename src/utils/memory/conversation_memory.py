"""Короткая память диалога на Supabase для LangChain.

Хранит историю в таблице `myaso.conversation_history` и возвращает
сообщения в формате LangChain.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from supabase import AClient

from src.config.database_constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CREATED_AT,
    COLUMN_MESSAGE,
    COLUMN_ROLE,
    TABLE_CONVERSATION_HISTORY,
)
from src.utils.async_mixin import AsyncMixin
from src.utils.supabase_client import get_supabase_client
from src.utils.rules import get_rule_as_int

logger = logging.getLogger(__name__)


_ROLE_TO_LC: Dict[str, type[BaseMessage]] = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}


def _to_role(message: BaseMessage) -> str:
    """Преобразует LangChain сообщение в роль для БД."""
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, ToolMessage):
        return "tool"
    return getattr(message, "role", "user")


def _from_role(role: str, content: str) -> BaseMessage:
    """Преобразует роль из БД в LangChain сообщение."""
    role = (role or "").lower()
    msg_cls = _ROLE_TO_LC.get(role, HumanMessage)
    if msg_cls is ToolMessage:
        return ToolMessage(content=content, tool_call_id=None)
    return msg_cls(content=content)


class SupabaseConversationMemory(AsyncMixin, BaseChatMessageHistory):
    """Память диалога на Supabase."""

    def __init__(self, client_phone: str) -> None:
        super().__init__(client_phone)
        self.client_phone = client_phone
        self.supabase: AClient | None = None

    async def __ainit__(self, client_phone: str) -> None:
        """Асинхронная инициализация памяти.

        Args:
            client_phone: Номер телефона клиента
        """
        self.client_phone = client_phone
        self.supabase = await get_supabase_client()
        logger.info(f"[SupabaseConversationMemory] Инициализирована память для {client_phone}")

    async def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Добавляет список сообщений в историю."""
        if not messages:
            return

        if self.supabase is None:
            logger.error(f"[SupabaseConversationMemory.add_messages] Supabase client не инициализирован для {self.client_phone}")
            raise RuntimeError("Supabase client is not initialized. Make sure to await the memory object after creation.")

        rows: List[Dict[str, Any]] = []
        for m in messages:
            rows.append(
                {
                    COLUMN_CLIENT_PHONE: self.client_phone,
                    COLUMN_ROLE: _to_role(m),
                    COLUMN_MESSAGE: m.content,
                }
            )

        try:
            logger.info(f"[SupabaseConversationMemory.add_messages] Сохранение {len(rows)} сообщений для {self.client_phone}")
            result = await self.supabase.table(TABLE_CONVERSATION_HISTORY).insert(rows).execute()
            logger.info(f"[SupabaseConversationMemory.add_messages] Успешно сохранено {len(rows)} сообщений для {self.client_phone}")
        except Exception as e:
            logger.error(f"[SupabaseConversationMemory.add_messages] Ошибка при сохранении сообщений для {self.client_phone}: {e}", exc_info=True)
            raise

    async def clear(self) -> None:
        """Удаляет историю для указанного `client_phone`."""
        assert self.supabase is not None, "Supabase client is not initialized"
        await (
            self.supabase.table(TABLE_CONVERSATION_HISTORY)
            .delete()
            .eq(COLUMN_CLIENT_PHONE, self.client_phone)
            .execute()
        )

    async def get_messages(self) -> List[BaseMessage]:
        """Возвращает сообщения в формате LangChain (по возрастанию времени).
        
        Ограничивает количество сообщений согласно MAX_HISTORY_MESSAGES из БД.
        """
        assert self.supabase is not None, "Supabase client is not initialized"
        
        try:
            max_history = await get_rule_as_int("MAX_HISTORY_MESSAGES")
        except Exception as e:
            logger.warning(f"[SupabaseConversationMemory] Не удалось загрузить MAX_HISTORY_MESSAGES из БД, используем 10: {e}")
            max_history = 10
        
        resp = (
            await self.supabase.table(TABLE_CONVERSATION_HISTORY)
            .select("*")
            .eq(COLUMN_CLIENT_PHONE, self.client_phone)
            .order(COLUMN_CREATED_AT, desc=False)
            .limit(max_history)
            .execute()
        )
        data: Iterable[Dict[str, Any]] = getattr(resp, "data", [])
        return [_from_role(r.get(COLUMN_ROLE, "user"), r.get(COLUMN_MESSAGE, "")) for r in data]

    async def load_memory_variables(
        self, inputs: Dict[str, Any] | None = None, *, return_messages: bool = True
    ) -> Dict[str, Any]:
        """Совместимость с ConversationBufferMemory.

        - Если `return_messages=True` — вернёт список `BaseMessage` в ключе `history`.
        - Иначе — объединённую текстовую стенограмму в ключе `history`.
        """
        msgs = await self.get_messages()
        if return_messages:
            return {"history": msgs}
        lines: List[str] = []
        for m in msgs:
            role = _to_role(m)
            lines.append(f"{role}: {m.content}")
        return {"history": "\n".join(lines)}
