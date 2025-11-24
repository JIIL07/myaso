"""Pydantic модель для промптов."""

from typing import Optional

from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """Модель промпта из таблицы myaso.prompts."""

    topic: str = Field(..., description="Тема промпта")
    prompt: Optional[str] = Field(None, description="Текст промпта")

    class Config:
        """Конфигурация модели."""

        from_attributes = True
        populate_by_name = True

