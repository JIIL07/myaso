from typing import Any

from src.services.memory.conversation_memory import SupabaseConversationMemory


def is_memory_initialized(memory: Any) -> bool:
    if isinstance(memory, SupabaseConversationMemory):
        return memory.supabase is not None
    return hasattr(memory, "async_initialized") and memory.async_initialized
