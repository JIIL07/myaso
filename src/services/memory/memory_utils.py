"""Утилиты для работы с памятью."""

from __future__ import annotations

from typing import Any


def is_memory_initialized(memory: Any) -> bool:
    """Проверяет, инициализирована ли память.

    Args:
        memory: Объект памяти (должен иметь атрибут async_initialized)

    Returns:
        True если память инициализирована, False иначе
    """
    return hasattr(memory, 'async_initialized') and memory.async_initialized

