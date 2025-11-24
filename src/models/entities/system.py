"""Pydantic модель для системных переменных."""

from typing import Optional

from pydantic import BaseModel, Field


class System(BaseModel):
    """Модель системной переменной из таблицы myaso.system."""

    topic: str = Field(..., description="Ключ системной переменной")
    value: Optional[str] = Field(None, description="Значение системной переменной")

    class Config:
        """Конфигурация модели."""

        from_attributes = True
        populate_by_name = True

