"""Модели для входящих запросов."""

import re

from pydantic import BaseModel, Field, field_validator

from src.toolkit import normalize_and_validate_phone


class ClientPhoneRequest(BaseModel):
    """Базовая модель с нормализованным номером телефона."""

    client_phone: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Номер телефона клиента",
    )

    @field_validator("client_phone")
    @classmethod
    def validate_client_phone(cls, v: str) -> str:
        """Валидирует и нормализует номер телефона."""
        return normalize_and_validate_phone(v)


class InitConversationRequest(ClientPhoneRequest):
    """Модель запроса для инициализации беседы."""


class UserMessageRequest(InitConversationRequest):
    """Модель запроса с сообщением пользователя."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст сообщения пользователя",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Валидирует и нормализует сообщение пользователя."""
        if not v or not v.strip():
            raise ValueError("Сообщение не может быть пустым")
        message = v.strip()
        message = re.sub(r"\s+", " ", message)
        return message


class ResetConversationRequest(ClientPhoneRequest):
    """Модель запроса для сброса истории беседы."""
