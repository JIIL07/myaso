from __future__ import annotations

import logging
from typing import Any, Iterable

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from supabase import AClient

from src.constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CREATED_AT,
    COLUMN_MESSAGE,
    COLUMN_ROLE,
    MAX_HISTORY_MESSAGES,
    TABLE_CONVERSATION_HISTORY,
)
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout

logger = logging.getLogger(__name__)

_ROLE_TO_LC: dict[str, type[BaseMessage]] = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
    "tool": ToolMessage,
}


def _to_role(message: BaseMessage) -> str:
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
    role = (role or "").lower()
    msg_cls = _ROLE_TO_LC.get(role, HumanMessage)
    if msg_cls is ToolMessage:
        return ToolMessage(content=content, tool_call_id=None)
    return msg_cls(content=content)


class SupabaseConversationMemory(BaseChatMessageHistory):
    """Conversation memory persisted in a Supabase table.

    Use the async factory method ``create()`` instead of ``__init__``:

        memory = await SupabaseConversationMemory.create(client_phone)
    """

    def __init__(self, client_phone: str, supabase: AClient) -> None:
        self.client_phone = client_phone
        self.supabase = supabase

    @classmethod
    async def create(cls, client_phone: str) -> SupabaseConversationMemory:
        """Async factory: initializes the Supabase client and returns a ready instance."""
        supabase = await get_supabase_client()
        return cls(client_phone=client_phone, supabase=supabase)

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    async def add_messages(self, messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> None:
        if not messages:
            return

        rows: list[dict[str, Any]] = [
            {
                COLUMN_CLIENT_PHONE: self.client_phone,
                COLUMN_ROLE: _to_role(m),
                COLUMN_MESSAGE: m.content,
            }
            for m in messages
        ]
        try:
            await execute_with_timeout(
                self.supabase.table(TABLE_CONVERSATION_HISTORY).insert(rows).execute(),
                operation_name="memory.add_messages(%s)" % self.client_phone,
            )
        except Exception as e:
            logger.error(
                "[Memory] Error saving messages for %s: %s",
                self.client_phone, e, exc_info=True,
            )
            raise

    async def clear(self) -> None:
        await execute_with_timeout(
            self.supabase.table(TABLE_CONVERSATION_HISTORY)
            .delete()
            .eq(COLUMN_CLIENT_PHONE, self.client_phone)
            .execute(),
            operation_name="memory.clear(%s)" % self.client_phone,
        )

    async def get_messages(self) -> list[BaseMessage]:
        resp = await execute_with_timeout(
            self.supabase.table(TABLE_CONVERSATION_HISTORY)
            .select("*")
            .eq(COLUMN_CLIENT_PHONE, self.client_phone)
            .order(COLUMN_CREATED_AT, desc=False)
            .limit(MAX_HISTORY_MESSAGES)
            .execute(),
            operation_name="memory.get_messages(%s)" % self.client_phone,
        )
        data: Iterable[dict[str, Any]] = getattr(resp, "data", [])
        return [_from_role(r.get(COLUMN_ROLE, "user"), r.get(COLUMN_MESSAGE, "")) for r in data]

    async def load_memory_variables(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        return_messages: bool = True,
    ) -> dict[str, Any]:
        msgs = await self.get_messages()
        if return_messages:
            return {"history": msgs}
        lines = [f"{_to_role(m)}: {m.content}" for m in msgs]
        return {"history": "\n".join(lines)}
