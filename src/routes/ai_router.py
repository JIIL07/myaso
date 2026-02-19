import logging

from fastapi import APIRouter, BackgroundTasks, Request
from src.utils.logger.masking import mask_phone
from src.entities import (
    ErrorResponse,
    InitConversationRequest,
    ResetConversationRequest,
    SuccessResponse,
    UserMessageRequest,
)
from src.services.ai.conversation import ConversationService
from src.services.ai.customer import ClientValidationError, CustomerService
from src.middleware.rate_limiter import limiter
from src.utils.responses import format_queue_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai")
conversation_service = ConversationService.instance()
customer_service = CustomerService()


@router.post(
    "/processConversation",
    status_code=200,
    response_model=SuccessResponse | ErrorResponse,
)
@limiter.limit("10/minute")
async def process_conversation(
    request: Request,
    body: UserMessageRequest,
):
    try:
        await customer_service.validate_client_for_conversation(body.client_phone)
    except ClientValidationError as e:
        logger.info(
            "[processConversation] Ignoring message from %s: %s",
            mask_phone(body.client_phone), e.message,
        )
        return ErrorResponse(success=False, error=e.message)

    result = await conversation_service.process_conversation(body)
    result = format_queue_response(result, body.client_phone, "processConversation")

    if result.get("success"):
        return SuccessResponse(success=True)
    else:
        return ErrorResponse(success=False, error=result.get("error", "Unknown error"))


@router.post("/initConversation", status_code=200, response_model=SuccessResponse | ErrorResponse)
@limiter.limit("5/minute")
async def init_conversation(
    request: Request,
    body: InitConversationRequest,
):
    result = await conversation_service.init_conversation(body)
    result = format_queue_response(result, body.client_phone, "initConversation")

    if result.get("success"):
        return SuccessResponse(success=True)
    else:
        return ErrorResponse(success=False, error=result.get("error", "Unknown error"))


@router.delete("/resetConversation", status_code=200, response_model=SuccessResponse)
@limiter.limit("5/minute")
async def reset_conversation(
    request: Request,
    body: ResetConversationRequest,
    background_tasks: BackgroundTasks,
):
    background_tasks.add_task(
        conversation_service.reset_conversation_async,
        client_phone=body.client_phone,
    )
    logger.info("[resetConversation] History reset started for %s", mask_phone(body.client_phone))
    return SuccessResponse(success=True)
