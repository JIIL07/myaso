"""Короткая память диалога на Supabase для LangChain.

Хранит историю в таблице `myaso.conversation_history` и возвращает
сообщения в формате LangChain.
"""

from __future__ import annotations

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

from src.utils.async_mixin import AsyncMixin
from src.utils.supabase_client import get_supabase_client


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
        super().__init__()
        self.client_phone = client_phone
        self.supabase: AClient | None = None

    async def __ainit__(self) -> None:
        self.supabase = await get_supabase_client()

    async def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Добавляет список сообщений в историю."""
        if not messages:
            return
        assert self.supabase is not None, "Supabase client is not initialized"

        rows: List[Dict[str, Any]] = []
        for m in messages:
            rows.append(
                {
                    "client_phone": self.client_phone,
                    "role": _to_role(m),
                    "message": m.content,
                }
            )
        await self.supabase.table("conversation_history").insert(rows).execute()

    async def clear(self) -> None:
        """Удаляет историю для указанного `client_phone`."""
        assert self.supabase is not None, "Supabase client is not initialized"
        await (
            self.supabase.table("conversation_history")
            .delete()
            .eq("client_phone", self.client_phone)
            .execute()
        )

    async def get_messages(self) -> List[BaseMessage]:
        """Возвращает сообщения в формате LangChain (по возрастанию времени)."""
        assert self.supabase is not None, "Supabase client is not initialized"
        resp = (
            await self.supabase.table("conversation_history")
            .select("*")
            .eq("client_phone", self.client_phone)
            .order("created_at", desc=False)
            .execute()
        )
        data: Iterable[Dict[str, Any]] = getattr(resp, "data", [])
        return [_from_role(r.get("role", "user"), r.get("message", "")) for r in data]

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
