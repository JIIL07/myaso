"""Модели для входящих запросов."""

import re

from pydantic import BaseModel, Field, validator


class InitConverastionRequest(BaseModel):
    """Модель запроса для инициализации беседы."""

    client_phone: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Номер телефона клиента",
    )
    topic: str = Field(
        default="Продать",
        min_length=1,
        max_length=100,
        description="Тема беседы",
    )

    @validator("client_phone")
    def validate_client_phone(cls, v):
        """Валидирует и нормализует номер телефона."""
        if not v or not v.strip():
            raise ValueError("Номер телефона не может быть пустым")
        return v.strip()

    @validator("topic")
    def validate_topic(cls, v):
        """Валидирует и нормализует тему беседы."""
        if not v or not v.strip():
            raise ValueError("Тема беседы не может быть пустой")
        return v.strip()


class UserMessageRequest(InitConverastionRequest):
    """Модель запроса с сообщением пользователя."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст сообщения пользователя",
    )

    @validator("message")
    def validate_message(cls, v):
        """Валидирует и нормализует сообщение пользователя."""
        if not v or not v.strip():
            raise ValueError("Сообщение не может быть пустым")
        message = v.strip()
        message = re.sub(r"\s+", " ", message)
        return message


class ResetConversationRequest(BaseModel):
    """Модель запроса для сброса истории беседы."""

    client_phone: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Номер телефона клиента",
    )

    @validator("client_phone")
    def validate_client_phone(cls, v):
        """Валидирует и нормализует номер телефона."""
        if not v or not v.strip():
            raise ValueError("Номер телефона не может быть пустым")
        return v.strip()

