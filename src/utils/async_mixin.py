from typing import Any, TypeVar, List, Dict
import re
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


def remove_markdown_symbols(text: str) -> str:
    """Удаляет markdown символы из текста для отправки в WhatsApp."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"#{1,6}\s+(.+?)$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def records_to_json(records: List[asyncpg.Record]) -> List[Dict[str, Any]]:
    """Конвертирует asyncpg.Record в список словарей (JSON-совместимый формат).

    Args:
        records: Список записей из asyncpg

    Returns:
        Список словарей с данными из записей
    """
    return [dict(record) for record in records]


def extract_product_titles_from_text(products_text: str) -> List[str]:
    """Извлекает названия товаров из текста с информацией о товарах.

    Парсит строки вида "Название: ..." из текста и возвращает список названий.

    Args:
        products_text: Текст с информацией о товарах

    Returns:
        Список названий товаров
    """
    if not products_text or "не найдены" in products_text.lower():
        return []

    titles = []
    lines = products_text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Название:"):
            title = line.replace("Название:", "").strip()
            if title and title != "Не указано":
                titles.append(title)

    return titles
