from pydantic import BaseModel, Field


class InitConverastionRequest(BaseModel):
    """Модель запроса для инициализации беседы."""

    client_phone: str
    topic: str = Field(default="Продать")


class UserMessageRequest(InitConverastionRequest):
    """Модель запроса с сообщением пользователя."""

    message: str


class ResetConversationRequest(BaseModel):
    """Модель запроса для сброса истории беседы."""

    client_phone: str
