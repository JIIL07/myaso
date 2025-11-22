"""Инструменты для управления контекстом агента."""

from __future__ import annotations

from langchain_core.tools import tool

# Глобальное состояние контекста (в продакшене можно использовать Redis или БД)
_agent_context: dict[str, dict[str, any]] = {}


def get_agent_context(client_phone: str) -> dict[str, any]:
    """Получает контекст агента для клиента.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Словарь с контекстом агента
    """
    if client_phone not in _agent_context:
        _agent_context[client_phone] = {
            "is_init_message": False,
            "require_photo": False,
        }
    return _agent_context[client_phone]


def clear_agent_context(client_phone: str) -> None:
    """Очищает контекст агента для клиента.

    Args:
        client_phone: Номер телефона клиента
    """
    if client_phone in _agent_context:
        del _agent_context[client_phone]


def create_context_tools(client_phone: str):
    """Создает инструменты для управления контекстом агента.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Список инструментов для управления контекстом
    """
    @tool
    def set_conversation_context(context_type: str) -> str:
        """Устанавливает тип контекста разговора.

        Используй этот инструмент когда нужно установить тип разговора:
        - "init" - для начального сообщения (init conversation)
        - "normal" - для обычного разговора

        Args:
            context_type: Тип контекста ("init" или "normal")

        Returns:
            Подтверждение установки контекста
        """
        context = get_agent_context(client_phone)
        if context_type.lower() == "init":
            context["is_init_message"] = True
            return "Контекст установлен: init conversation (начальное сообщение)"
        elif context_type.lower() == "normal":
            context["is_init_message"] = False
            return "Контекст установлен: normal conversation (обычный разговор)"
        else:
            return f"Неизвестный тип контекста: {context_type}. Используй 'init' или 'normal'"

    @tool
    def set_photo_requirement(require: bool) -> str:
        """Устанавливает требование наличия фотографий для поиска товаров.

        Используй этот инструмент когда нужно указать, что поиск должен возвращать
        только товары с фотографиями.

        Args:
            require: True если требуются только товары с фото, False если фото не обязательны

        Returns:
            Подтверждение установки требования
        """
        context = get_agent_context(client_phone)
        context["require_photo"] = require
        if require:
            return "Требование установлено: возвращать только товары с фотографиями"
        else:
            return "Требование снято: возвращать все товары независимо от наличия фото"

    @tool
    def get_conversation_context() -> str:
        """Получает текущий контекст разговора.

        Returns:
            Информация о текущем контексте
        """
        context = get_agent_context(client_phone)
        context_type = "init" if context["is_init_message"] else "normal"
        photo_req = "требуются" if context["require_photo"] else "не требуются"
        return f"Тип контекста: {context_type}, Фото: {photo_req}"

    return [set_conversation_context, set_photo_requirement, get_conversation_context]


def get_is_init_message(client_phone: str) -> bool:
    """Получает флаг is_init_message для клиента.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        True если это init conversation, False иначе
    """
    context = get_agent_context(client_phone)
    return context.get("is_init_message", False)


def get_require_photo(client_phone: str) -> bool:
    """Получает флаг require_photo для клиента.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        True если требуются только товары с фото, False иначе
    """
    context = get_agent_context(client_phone)
    return context.get("require_photo", False)

