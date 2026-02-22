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

from src.constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CREATED_AT,
    COLUMN_MESSAGE,
    COLUMN_ROLE,
    MAX_HISTORY_MESSAGES,
    TABLE_CONVERSATION_HISTORY,
)
from src.services.database.database import get_pool
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


class PostgresConversationMemory(BaseChatMessageHistory):
    """Conversation memory persisted in PostgreSQL.

    Use the async factory method ``create()`` instead of ``__init__``:

        memory = await PostgresConversationMemory.create(client_phone)
    """

    def __init__(self, client_phone: str) -> None:
        self.client_phone = client_phone
        self.async_initialized = True

    @classmethod
    async def create(cls, client_phone: str) -> PostgresConversationMemory:
        """Async factory: initializes DB access and returns a ready instance."""
        # Warm up the shared pool once to fail-fast on misconfiguration.
        await get_pool()
        return cls(client_phone=client_phone)

    # ------------------------------------------------------------------
    # BaseChatMessageHistory interface
    # ------------------------------------------------------------------

    async def add_messages(self, messages: list[BaseMessage] | tuple[BaseMessage, ...]) -> None:
        if not messages:
            return

        rows: list[tuple[str, str, Any]] = [
            (
                self.client_phone,
                _to_role(m),
                m.content,
            )
            for m in messages
        ]
        try:
            pool = await get_pool()
            async with pool.acquire() as conn:
                await execute_with_timeout(
                    conn.executemany(
                        """
                        INSERT INTO myaso.%s (%s, %s, %s)
                        VALUES ($1, $2, $3)
                        """
                        % (
                            TABLE_CONVERSATION_HISTORY,
                            COLUMN_CLIENT_PHONE,
                            COLUMN_ROLE,
                            COLUMN_MESSAGE,
                        ),
                        rows,
                    ),
                    operation_name="memory.add_messages(%s)" % self.client_phone,
                )
        except Exception as e:
            logger.error(
                "[Memory] Error saving messages for %s: %s",
                self.client_phone, e, exc_info=True,
            )
            raise

    async def clear(self) -> None:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await execute_with_timeout(
                conn.execute(
                    """
                    DELETE FROM myaso.%s
                    WHERE %s = $1
                    """
                    % (TABLE_CONVERSATION_HISTORY, COLUMN_CLIENT_PHONE),
                    self.client_phone,
                ),
                operation_name="memory.clear(%s)" % self.client_phone,
            )

    async def get_messages(self) -> list[BaseMessage]:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await execute_with_timeout(
                conn.fetch(
                    """
                    SELECT *
                    FROM myaso.%s
                    WHERE %s = $1
                    ORDER BY %s ASC
                    LIMIT $2
                    """
                    % (
                        TABLE_CONVERSATION_HISTORY,
                        COLUMN_CLIENT_PHONE,
                        COLUMN_CREATED_AT,
                    ),
                    self.client_phone,
                    MAX_HISTORY_MESSAGES,
                ),
                operation_name="memory.get_messages(%s)" % self.client_phone,
            )
        data: Iterable[dict[str, Any]] = [dict(row) for row in rows]
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
