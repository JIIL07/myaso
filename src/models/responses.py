"""Модели для исходящих ответов."""

from typing import Optional, Dict, Any
from pydantic import BaseModel


class ClientProfileResponse(BaseModel):
    """Модель ответа с профилем клиента."""

    client_phone: str
    profile: str
    message_count: int
    last_order: Optional[Dict[str, Any]] = None
    status: str

