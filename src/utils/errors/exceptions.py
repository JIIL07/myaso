"""Базовые классы исключений для унификации обработки ошибок."""


class DatabaseError(Exception):
    """Базовый класс для ошибок базы данных.
    
    Используется для всех ошибок, связанных с операциями БД.
    """
    pass


class RecordNotFoundError(DatabaseError):
    """Исключение для случая, когда запись не найдена в БД.
    
    Args:
        entity_type: Тип сущности (например, "Client", "Product")
        identifier: Идентификатор сущности (например, номер телефона или ID)
        message: Дополнительное сообщение об ошибке
    """
    
    def __init__(
        self,
        entity_type: str,
        identifier: str,
        message: str | None = None,
    ) -> None:
        self.entity_type = entity_type
        self.identifier = identifier
        if message is None:
            message = f"{entity_type} not found: {identifier}"
        super().__init__(message)


class DatabaseConnectionError(DatabaseError):
    """Исключение для ошибок подключения к БД.
    
    Args:
        message: Сообщение об ошибке
        original_error: Исходное исключение (если есть)
    """
    
    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        self.original_error = original_error
        super().__init__(message)


class DatabaseTimeoutError(DatabaseError):
    """Исключение для timeout при выполнении запросов к БД.
    
    Args:
        operation: Название операции, которая превысила timeout
        timeout: Значение timeout в секундах
    """
    
    def __init__(
        self,
        operation: str,
        timeout: float,
    ) -> None:
        self.operation = operation
        self.timeout = timeout
        message = f"Database operation '{operation}' exceeded timeout of {timeout}s"
        super().__init__(message)


class AgentError(Exception):
    """Базовый класс для ошибок агента.
    
    Args:
        message: Сообщение об ошибке
        details: Дополнительные детали ошибки (словарь)
    """
    
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AgentTimeoutError(AgentError):
    """Агент превысил максимальное время выполнения."""
    pass


class AgentExecutionError(AgentError):
    """Ошибка при выполнении агента."""
    pass
