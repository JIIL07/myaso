"""Модели для исходящих ответов."""

from typing import Any, Optional

from pydantic import BaseModel


class SuccessResponse(BaseModel):
    """Модель успешного ответа."""

    success: bool = True


class ErrorResponse(BaseModel):
    """Модель ответа с ошибкой."""

    success: bool = False
    error: str
    details: Optional[Any] = None


class TestResponse(BaseModel):
    """Модель ответа для тестовых эндпоинтов."""

    success: bool
    response_text: Optional[str] = None
    error: Optional[str] = None
