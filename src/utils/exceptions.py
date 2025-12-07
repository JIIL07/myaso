"""Кастомные исключения для приложения Myaso."""

from __future__ import annotations


class MyasoBaseException(Exception):
    """Базовое исключение для всех ошибок приложения Myaso."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Инициализация исключения.

        Args:
            message: Сообщение об ошибке
            details: Дополнительные детали об ошибке
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DatabaseError(MyasoBaseException):
    """Ошибка при работе с базой данных."""

    pass


class ValidationError(MyasoBaseException):
    """Ошибка валидации данных."""

    pass


class ConfigurationError(MyasoBaseException):
    """Ошибка конфигурации приложения."""

    pass


class AgentError(MyasoBaseException):
    """Ошибка при работе агента."""

    pass


class AgentTimeoutError(AgentError):
    """Ошибка при таймауте агента."""

    pass


class AgentExecutionError(AgentError):
    """Ошибка при выполнении агента."""

    pass


class WhatsAppError(MyasoBaseException):
    """Ошибка при работе с WhatsApp API."""

    pass


class RuleNotFoundError(ConfigurationError):
    """Ошибка при отсутствии правила в базе данных."""

    pass

