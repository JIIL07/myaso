"""Pydantic модель для контекста агента."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AgentContext(BaseModel):
    """Модель контекста агента из таблицы myaso.agent_context."""

    client_phone: str = Field(..., description="Номер телефона клиента")
    context_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Данные контекста в формате JSONB"
    )
    created_at: Optional[datetime] = Field(None, description="Дата и время создания записи")
    updated_at: Optional[datetime] = Field(None, description="Дата и время последнего обновления")

    class Config:
        """Конфигурация модели."""

        from_attributes = True
        populate_by_name = True

