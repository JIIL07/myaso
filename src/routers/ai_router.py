import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks
from supabase import AClient

from src.agents.factory import AgentFactory
from src.config.database_constants import (
    COLUMN_CLIENT_PHONE,
    COLUMN_CREATED_AT,
    COLUMN_DESTINATION,
    COLUMN_PRICE_OUT,
    COLUMN_TITLE,
    COLUMN_WEIGHT_KG,
    TABLE_CONVERSATION_HISTORY,
    TABLE_ORDERS,
)
from src.config.messages_constants import (
    ERROR_MESSAGE_INVALID_PHONE,
    ERROR_MESSAGE_WHATSAPP_FAILED,
    PROMPT_TOPIC_WELCOME_MESSAGE,
    SYSTEM_VALUE_PRICELIST,
)
from src.config.settings import settings
from src.database.queries.clients_queries import (
    get_client_by_phone,
    get_client_send_message,
)
from src.database.queries.history_queries import get_conversation_history_count
from src.models import (
    ClientProfileResponse,
    InitConverastionRequest,
    ResetConversationRequest,
    UserMessageRequest,
)
from src.services.whatsapp_service import send_image, send_message
from src.utils import get_supabase_client, remove_markdown_symbols
from src.utils.memory import SupabaseConversationMemory
from src.utils.memory_utils import is_memory_initialized
from src.utils.phone_validator import normalize_phone
from src.utils.prompts import get_prompt, get_system_value
from src.utils.validation_utils import validate_and_normalize_phone
from src.routers.ai_router_helpers import (
    prepare_user_input_for_agent,
    send_pricelist_if_needed,
    send_agent_response,
    send_error_message,
)
from src.utils.queue import send_delayed_message

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai")


async def process_conversation_background(request: UserMessageRequest):
    """Обрабатывает запрос пользователя в фоновом режиме.

    Args:
        request: Запрос с сообщением пользователя и номером телефона
    """

    try:
        memory = await SupabaseConversationMemory(request.client_phone)

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        user_input_with_tool_signature = await prepare_user_input_for_agent(
            request.message,
            is_init=False,
            topic=request.topic
        )

        response_text = await agent.run(
            user_input=user_input_with_tool_signature,
            client_phone=request.client_phone,
            topic=request.topic,
            endpoint_name="processConversation",
        )

        try:
            success = await send_message(
                request.client_phone,
                remove_markdown_symbols(response_text),
            )
            if not success:
                logger.error(
                    f"[processConversation] Не удалось отправить сообщение в WhatsApp для {request.client_phone} (send_message вернул False)"
                )
        except Exception as e:
            logger.error(
                f"[processConversation] ОШИБКА: Ошибка отправки в WhatsApp для {request.client_phone}: {e}",
                exc_info=True,
            )

        return {"success": True}

    except Exception as e:
        logger.error(
            f"[processConversation] Ошибка обработки для {request.client_phone}: {e}",
            exc_info=True,
        )
        send_error_message(request.client_phone, "processConversation")

        return {"success": False}


@router.post("/processConversation", status_code=200)
async def process_conversation(
    request: UserMessageRequest, background_tasks: BackgroundTasks
):
    """Обрабатывает запрос пользователя и запускает фоновую задачу.

    Args:
        request: Запрос с сообщением пользователя
        background_tasks: Фоновые задачи FastAPI

    Returns:
        Словарь с результатом успешного запуска задачи
    """
    normalized_phone, validation_result = validate_and_normalize_phone(request.client_phone)
    if not validation_result["success"]:
        logger.error(
            f"[processConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return validation_result

    request.client_phone = normalized_phone
    
    # Проверка наличия клиента в БД
    client = await get_client_by_phone(request.client_phone)
    if client is None:
        logger.info(
            f"[processConversation] Игнорируем сообщение от {request.client_phone}: клиент не найден в БД"
        )
        return {"success": False, "error": "Client not found in database"}
    
    # Проверка, был ли вызван initConversation (есть ли записи в conversation_history)
    history_count = await get_conversation_history_count(request.client_phone)
    if history_count == 0:
        logger.info(
            f"[processConversation] Игнорируем сообщение от {request.client_phone}: разговор не был инициализирован через initConversation"
        )
        return {"success": False, "error": "Conversation not initialized"}
    
    # Проверка send_message
    send_message_enabled = await get_client_send_message(request.client_phone)
    if not send_message_enabled:
        logger.info(
            f"[processConversation] Игнорируем сообщение от {request.client_phone}: отправка сообщений отключена (send_message=false)"
        )
        return {"success": False, "error": "Message sending disabled"}
    
    # Отправка сообщения в очередь PGMQ с задержкой 15 минут
    msg_id = await send_delayed_message(
        client_phone=request.client_phone,
        message=request.message,
        topic=request.topic,
    )
    
    if msg_id is None:
        logger.error(
            f"[processConversation] Не удалось добавить сообщение в очередь для {request.client_phone}"
        )
        return {"success": False, "error": "Failed to queue message"}
    
    logger.info(
        f"[processConversation] Сообщение добавлено в очередь для {request.client_phone}, msg_id={msg_id}"
    )
    return {"success": True}


async def init_conversation_background(request: InitConverastionRequest):
    """Инициализирует новую беседу с клиентом в фоновом режиме.

    Args:
        request: Запрос с номером телефона клиента и темой беседы
    """

    try:
        memory = await SupabaseConversationMemory(request.client_phone)
        
        if not is_memory_initialized(memory):
            logger.warning(f"[initConversation] Память не инициализирована для {request.client_phone}, инициализируем...")
            await memory.__ainit__(request.client_phone)
    
        await memory.clear()

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        welcome_input = await get_prompt(PROMPT_TOPIC_WELCOME_MESSAGE)
        
        if not welcome_input:
            logger.warning(
                f"[initConversation] Промпт для topic '{PROMPT_TOPIC_WELCOME_MESSAGE}' не найден в БД для {request.client_phone}"
            )
            welcome_input = ""
        else:
            welcome_input = welcome_input.replace("{client_phone}", request.client_phone)

        # Подготавливаем ввод с инструкциями из БД
        prepared_input = await prepare_user_input_for_agent(
            welcome_input,
            is_init=True,
            topic=request.topic
        )

        response_text = await agent.run(
            user_input=prepared_input,
            client_phone=request.client_phone,
            topic=request.topic,
            endpoint_name="initConversation",
        )

        # Отправка сообщения
        message_sent = await send_agent_response(
            request.client_phone,
            response_text,
            "initConversation"
        )
        
        if not message_sent:
            raise Exception("Не удалось отправить сообщение клиенту")

        # Отправка прайс-листа (только если сообщение отправлено успешно)
        if message_sent:
            await send_pricelist_if_needed(request.client_phone)

        return {"success": True}

    except Exception as e:
        logger.error(
            f"[initConversation] Критическая ошибка для {request.client_phone}: {e}",
            exc_info=True,
        )
        send_error_message(request.client_phone, "initConversation")

        return {"success": False}


@router.post("/initConversation", status_code=200)
async def init_conversation(
    request: InitConverastionRequest, background_tasks: BackgroundTasks
):
    """Инициализирует новую беседу и запускает фоновую задачу.

    Args:
        request: Запрос с номером телефона и темой беседы
        background_tasks: Фоновые задачи FastAPI

    Returns:
        Словарь с результатом успешного запуска задачи
    """
    normalized_phone, validation_result = validate_and_normalize_phone(request.client_phone)
    if not validation_result["success"]:
        logger.error(
            f"[initConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return validation_result

    request.client_phone = normalized_phone
    background_tasks.add_task(init_conversation_background, request)
    return {"success": True}


@router.get("/getProfile", response_model=ClientProfileResponse, status_code=200)
async def get_profile(client_phone: str):
    """Получает профиль клиента по номеру телефона.

    Args:
        client_phone: Номер телефона клиента

    Returns:
        Модель с профилем клиента, количеством сообщений и последним заказом
    """
    client_phone = normalize_phone(client_phone)

    try:
        from src.agents.tools import get_client_profile
        profile_text = await get_client_profile.ainvoke({"phone": client_phone})
    except Exception:
        profile_text = "Профиль клиента не найден в базе данных."

    message_count = 0
    last_order: Optional[Dict[str, Any]] = None
    supabase: AClient | None = None

    try:
        supabase = await get_supabase_client()

        history_resp = (
            await supabase.table(TABLE_CONVERSATION_HISTORY)
            .select("*")
            .eq(COLUMN_CLIENT_PHONE, client_phone)
            .execute()
        )
        message_count = len(history_resp.data) if history_resp.data else 0

        from src.database.queries.orders_queries import get_last_order
        order = await get_last_order(client_phone)
        if order:
            last_order = {
                COLUMN_TITLE: order.title,
                COLUMN_CREATED_AT: order.created_at.isoformat() if order.created_at else None,
                COLUMN_DESTINATION: order.destination,
                COLUMN_PRICE_OUT: order.price_out,
                COLUMN_WEIGHT_KG: order.weight_kg,
            }
    except Exception:
        pass

    status = "active" if (message_count > 0 or last_order is not None) else "new"

    return ClientProfileResponse(
        client_phone=client_phone,
        profile=profile_text,
        message_count=message_count,
        last_order=last_order,
        status=status,
    )


async def reset_conversation_background(request: ResetConversationRequest):
    """Сбрасывает историю беседы для клиента в фоновом режиме.

    Args:
        request: Запрос с номером телефона клиента
    """

    try:
        memory = await SupabaseConversationMemory(request.client_phone)
        await memory.clear()

        return {"success": True}

    except Exception as e:
        logger.error(
            f"[resetConversation] Ошибка для {request.client_phone}: {e}", exc_info=True
        )
        return {"success": False}


@router.delete("/resetConversation", status_code=200)
async def reset_conversation(
    request: ResetConversationRequest, background_tasks: BackgroundTasks
):
    """Сбрасывает историю беседы и запускает фоновую задачу.

    Args:
        request: Запрос с номером телефона клиента
        background_tasks: Фоновые задачи FastAPI

    Returns:
        Словарь с результатом успешного запуска задачи
    """
    normalized_phone, validation_result = validate_and_normalize_phone(request.client_phone)
    if not validation_result["success"]:
        logger.error(
            f"[resetConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return validation_result

    request.client_phone = normalized_phone
    background_tasks.add_task(reset_conversation_background, request)
    return {"success": True}