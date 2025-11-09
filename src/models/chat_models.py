"""Модели для чата и памяти.

Модели для работы с историей диалогов и памятью.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ConversationMessage(BaseModel):
    """Модель сообщения в диалоге."""

    role: str
    content: str
    timestamp: Optional[str] = None


class ConversationHistory(BaseModel):
    """Модель истории диалога."""

    client_phone: str
    messages: List[ConversationMessage]
    total_messages: int


class LangFuseTraceResponse(BaseModel):
    """Модель ответа с трейсом LangFuse."""

    trace_id: Optional[str] = None
    timestamp: Optional[str] = None
    input: Dict[str, Any] = {}
    output: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}


class ConversationHistoryResponse(BaseModel):
    """Модель ответа с историей диалога из LangFuse."""

    phone: str
    total_conversations: int
    days: int
    history: List[LangFuseTraceResponse]
    error: Optional[str] = None

