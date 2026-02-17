"""Исключения для сервисов AI."""


class ClientValidationError(Exception):
    """Ошибка валидации клиента."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ConversationError(Exception):
    """Ошибка обработки разговора."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
