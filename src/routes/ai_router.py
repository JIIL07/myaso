"""API-роуты для работы с AI-агентом."""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from src.utils.logger.masking import mask_phone
from src.entities import (
    ErrorResponse,
    InitConversationRequest,
    ResetConversationRequest,
    SuccessResponse,
    TestResponse,
    UserMessageRequest,
)
from src.services.ai.conversation import ConversationService
from src.services.ai.customer import ClientValidationError, CustomerService
from src.middleware.rate_limiter import limiter
from src.utils.responses import format_test_response, format_queue_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai")
conversation_service = ConversationService()
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
    """Обрабатывает сообщение пользователя или ставит его в очередь."""
    try:
        await customer_service.validate_client_for_conversation(body.client_phone)
    except ClientValidationError as e:
        logger.info(
            f"[processConversation] Игнорируем сообщение от {mask_phone(body.client_phone)}: {e.message}"
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
    """Инициализирует беседу с клиентом или ставит задачу в очередь."""
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
    """Сбрасывает историю беседы и запускает задачу в фоне."""
    background_tasks.add_task(
        conversation_service.reset_conversation_async,
        client_phone=body.client_phone,
    )

    logger.info(f"[resetConversation] Сброс истории начат для {mask_phone(body.client_phone)}")
    return SuccessResponse(success=True)


@router.post(
    "/test/process",
    status_code=200,
    response_model=TestResponse,
)
@limiter.limit("30/minute")
async def test_process_conversation(
    request: Request,
    body: UserMessageRequest,
):
    """Тестовая обработка сообщения без уведомлений и очереди."""
    return await format_test_response(
        conversation_service.process_conversation_test,
        body,
        "test/process",
    )


@router.post(
    "/test/init",
    status_code=200,
    response_model=TestResponse,
)
@limiter.limit("30/minute")
async def test_init_conversation(
    request: Request,
    body: InitConversationRequest,
):
    """Тестовая инициализация беседы без отправки уведомлений."""
    return await format_test_response(
        conversation_service.init_conversation_test,
        body,
        "test/init",
    )


@router.get("/dev/queue", status_code=200)
async def get_queue_status(request: Request):
    """Возвращает статус очереди и агента."""
    status = conversation_service.get_queue_status()
    return status
