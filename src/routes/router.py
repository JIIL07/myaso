"""API роуты для получения сообщений от сторонних API."""
import logging

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel, Field

from src.entities import ErrorResponse, SuccessResponse, UserMessageRequest
from src.queries.clients_queries import get_client_by_phone
from src.routes.ai_router import process_conversation
from src.utils.validators.phone_validator import normalize_phone, validate_phone

logger = logging.getLogger(__name__)
router = APIRouter()


class ExternalMessageRequest(BaseModel):
    """Модель запроса от стороннего API."""

    phone: str = Field(..., description="Номер телефона клиента")
    message: str = Field(..., description="Текст сообщения")


@router.post("/get_message", status_code=200, response_model=SuccessResponse | ErrorResponse)
async def get_message(
    request: ExternalMessageRequest,
    background_tasks: BackgroundTasks,
):
    """Получает сообщение от стороннего API, проверяет пользователя и редиректит на /ai/processConversation.

    Args:
        request: Запрос с сообщением от стороннего API
        background_tasks: Фоновые задачи FastAPI

    Returns:
        Результат обработки сообщения (редирект на /ai/processConversation)
    """
    normalized_phone = normalize_phone(request.phone)
    
    if not validate_phone(normalized_phone):
        logger.warning(f"[get_message] Невалидный номер телефона: {request.phone}")
        return ErrorResponse(success=False, error="Invalid phone number format")

    client = await get_client_by_phone(normalized_phone)
    if client is None:
        logger.warning(f"[get_message] Пользователь не найден в БД: {normalized_phone}")
        return ErrorResponse(success=False, error="User not found in database")

    logger.info(f"[get_message] Пользователь найден в БД: {normalized_phone}, редирект на /ai/processConversation")

    user_message_request = UserMessageRequest(
        client_phone=normalized_phone,
        message=request.message,
    )

    return await process_conversation(
        request=user_message_request,
        background_tasks=background_tasks,
    )
