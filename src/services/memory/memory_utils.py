from typing import Any

from src.services.memory.conversation_memory import PostgresConversationMemory


def is_memory_initialized(memory: Any) -> bool:
    if isinstance(memory, PostgresConversationMemory):
        return bool(getattr(memory, "async_initialized", False))
    return hasattr(memory, "async_initialized") and memory.async_initialized
