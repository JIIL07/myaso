"""Модели для входящих запросов."""

import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.utils.validators.phone_validator import normalize_phone, validate_phone


class InitConversationRequest(BaseModel):
    """Модель запроса для инициализации беседы."""

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
        if not v or not v.strip():
            raise ValueError("Номер телефона не может быть пустым")
        
        normalized = normalize_phone(v.strip())
        
        if not validate_phone(normalized):
            raise ValueError("Неверный формат номера телефона")
        
        return normalized


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


class ResetConversationRequest(BaseModel):
    """Модель запроса для сброса истории беседы."""

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
        if not v or not v.strip():
            raise ValueError("Номер телефона не может быть пустым")
        
        normalized = normalize_phone(v.strip())
        
        if not validate_phone(normalized):
            raise ValueError("Неверный формат номера телефона")
        
        return normalized
