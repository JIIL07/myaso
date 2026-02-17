import re
from typing import Any, Dict, List, TypeVar

import asyncpg

T = TypeVar("T", bound="AsyncMixin")


class AsyncMixin:
    """Асинхронный миксин для поддержки асинхронной инициализации объектов.

    Позволяет классам, использующим этот миксин, иметь асинхронную инициализацию
    через метод `__ainit__`. Использование: `obj = await AsyncClass(...)`

    Пример:
        class MyClass(AsyncMixin):
            async def __ainit__(self, value: str):
                self.value = await some_async_op(value)

        obj = await MyClass("test")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Сохраняет аргументы для последующей асинхронной инициализации.

        Args:
            *args: Позиционные аргументы для передачи в __ainit__
            **kwargs: Именованные аргументы для передачи в __ainit__
        """
        self.__storedargs = args, kwargs
        self.async_initialized = False

    async def __ainit__(self, *args: Any, **kwargs: Any) -> None:
        """Асинхронная инициализация объекта.

        Переопределите этот метод в наследниках для выполнения
        асинхронных операций инициализации.

        Args:
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы
        """
        pass

    async def __initobj(self: T) -> T:
        """Внутренний метод для асинхронной инициализации.

        Args:
            self: Экземпляр класса

        Returns:
            Инициализированный экземпляр класса

        Raises:
            AssertionError: Если объект уже был инициализирован
        """
        assert not self.async_initialized
        self.async_initialized = True
        await self.__ainit__(*self.__storedargs[0], **self.__storedargs[1])
        return self

    def __await__(self) -> Any:
        """Поддержка await для асинхронной инициализации.

        Returns:
            Корутина для инициализации объекта
        """
        return self.__initobj().__await__()



