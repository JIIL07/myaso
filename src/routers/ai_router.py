from fastapi import APIRouter, BackgroundTasks
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
from src.models import (
    UserMessageRequest,
    InitConverastionRequest,
    ResetConversationRequest,
    ClientProfileResponse,
)
from src.agents.factory import AgentFactory
from src.utils import remove_markdown_symbols
from src.utils.memory import SupabaseConversationMemory
from src.utils.phone_validator import normalize_phone, validate_phone
from src.config.settings import settings
from src.utils import get_supabase_client
from src.services.whatsapp_service import send_message
from supabase import AClient

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

        response_text = await agent.run(
            user_input=request.message,
            client_phone=request.client_phone,
            topic=request.topic,
            endpoint_name="processConversation",
        )


        try:
            await send_message(
                request.client_phone,
                remove_markdown_symbols(response_text),
            )
        except Exception as e:
            logger.error(f"ОШИБКА: Ошибка отправки в WhatsApp для {request.client_phone}: {e}")

        return {"success": True}

    except Exception as e:
        logger.error(
            f"[processConversation] Ошибка обработки для {request.client_phone}: {e}",
            exc_info=True,
        )
        try:
            await send_message(
                request.client_phone,
                "Что-то вотсап барахлит 😔. Напишите позже, пожалуйста!",
            )
        except Exception:
            pass

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
    normalized_phone = normalize_phone(request.client_phone)
    if not validate_phone(normalized_phone):
        logger.error(
            f"[processConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return {"success": False, "error": "Invalid phone number"}

    request.client_phone = normalized_phone
    background_tasks.add_task(process_conversation_background, request)
    return {"success": True}


async def init_conversation_background(request: InitConverastionRequest):
    """Инициализирует новую беседу с клиентом в фоновом режиме.

    Args:
        request: Запрос с номером телефона клиента и темой беседы
    """

    try:
        memory = await SupabaseConversationMemory(request.client_phone)
        await memory.clear()

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        welcome_input = f"""Сформируй короткое дружелюбное приветствие для клиента.
Тема диалога: {request.topic}
Для формирования приветствия:
1. Получи профиль клиента (номер телефона: {request.client_phone})
2. Получи товары по теме диалога "{request.topic}" используя generate_sql_from_text + execute_sql_request (text2sql инструменты)
   ВАЖНО: Это init_conversation, поэтому предпочтительно использовать text2sql инструменты для поиска товаров
3. Если есть товары с фотографиями, отправь их клиенту
Поприветствуй дружелюбно со смайликами, будь позитивным и энергичным. Предложи помощь и ненавязчиво уточни запрос."""

        response_text = await agent.run(
            user_input=welcome_input,
            client_phone=request.client_phone,
            topic=request.topic,
            is_init_message=True,
            endpoint_name="initConversation",
        )

        try:
            await send_message(
                request.client_phone,
                remove_markdown_symbols(response_text),
            )
        except Exception as send_error:
            logger.error(
                f"[initConversation] Ошибка отправки сообщения в WhatsApp для {request.client_phone}: {send_error}",
                exc_info=True,
            )
            raise


        return {"success": True}

    except Exception as e:
        logger.error(
            f"[initConversation] Критическая ошибка для {request.client_phone}: {e}",
            exc_info=True,
        )
        try:
            await send_message(
                request.client_phone,
                "Что-то вотсап барахлит 😔. Напишите позже, пожалуйста!",
            )
        except Exception as send_error:
            logger.error(
                f"[initConversation] Ошибка отправки сообщения об ошибке: {send_error}"
            )

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
    normalized_phone = normalize_phone(request.client_phone)
    if not validate_phone(normalized_phone):
        logger.error(
            f"[initConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return {"success": False, "error": "Invalid phone number"}

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
            await supabase.table("conversation_history")
            .select("*")
            .eq("client_phone", client_phone)
            .execute()
        )
        message_count = len(history_resp.data) if history_resp.data else 0

        orders_resp = (
            await supabase.table("orders")
            .select("*")
            .eq("client_phone", client_phone)
            .order("created_at", desc=True)
            .execute()
        )
        orders = orders_resp.data if orders_resp.data else []
        if orders:
            o = orders[0]
            last_order = {
                "title": o.get("title"),
                "created_at": o.get("created_at"),
                "destination": o.get("destination"),
                "price_out": o.get("price_out"),
                "weight_kg": o.get("weight_kg"),
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
    normalized_phone = normalize_phone(request.client_phone)
    if not validate_phone(normalized_phone):
        logger.error(
            f"[resetConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return {"success": False, "error": "Invalid phone number"}


    request.client_phone = normalized_phone
    background_tasks.add_task(reset_conversation_background, request)
    return {"success": True}