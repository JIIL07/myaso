from typing import Any, TypeVar, List, Dict
import re
import asyncpg

T = TypeVar("T", bound="AsyncMixin")


class AsyncMixin:
    """Асинхронный миксин для поддержки асинхронной инициализации объектов."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __ainit__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __initobj(self: T) -> T:
        assert not self.async_initialized
        self.async_initialized = True
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self):
        return self.__initobj().__await__()


def remove_markdown_symbols(text: str) -> str:
    """Удаляет markdown символы из текста для отправки в WhatsApp."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'#{1,6}\s+(.+?)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()


def records_to_json(records: List[asyncpg.Record]) -> List[Dict[str, Any]]:
    """Конвертирует asyncpg.Record в список словарей (JSON-совместимый формат).
    
    Args:
        records: Список записей из asyncpg
        
    Returns:
        Список словарей с данными из записей
    """
    return [dict(record) for record in records]

