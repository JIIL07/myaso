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

    phone: str = Field(..., description="Client phone number")
    message: str = Field(..., description="Message text")


@router.post("/get_message", status_code=200, response_model=SuccessResponse | ErrorResponse)
async def get_message(
    request: ExternalMessageRequest,
    background_tasks: BackgroundTasks,
):
    normalized_phone = normalize_phone(request.phone)

    if not validate_phone(normalized_phone):
        logger.warning("[get_message] Invalid phone number: %s", request.phone)
        return ErrorResponse(success=False, error="Invalid phone number format")

    client = await get_client_by_phone(normalized_phone)
    if client is None:
        logger.warning("[get_message] User not found in DB: %s", normalized_phone)
        return ErrorResponse(success=False, error="User not found in database")

    logger.info("[get_message] User found in DB: %s, redirecting to /ai/processConversation", normalized_phone)

    user_message_request = UserMessageRequest(
        client_phone=normalized_phone,
        message=request.message,
    )

    return await process_conversation(
        request=user_message_request,
        background_tasks=background_tasks,
    )
