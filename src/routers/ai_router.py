from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime, timedelta
from src.schemas import (
    UserMessageRequest,
    InitConverastionRequest,
    ResetConversationRequest,
)
from src.config.constants import (
    HTTP_TIMEOUT_SECONDS,
)
from agents.factory import AgentFactory
from src.utils import remove_markdown_symbols
from src.utils.langchain_memory import SupabaseConversationMemory
from src.utils.phone_validator import normalize_phone, validate_phone
from src.config.settings import settings
from src.utils.supabase_client import get_supabase_client
from supabase import AClient
import httpx

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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                await client.post(
                    settings.whatsapp.send_message_url,
                    json={
                        "recipient": request.client_phone,
                        "message": remove_markdown_symbols(response_text),
                    },
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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                await client.post(
                    settings.whatsapp.send_message_url,
                    json={
                        "recipient": request.client_phone,
                        "message": "Что-то вотсап барахлит 😔. Напишите позже, пожалуйста!",
                    },
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
        logger.warning(
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
2. Получи товары по теме диалога "{request.topic}" используя подходящие инструменты
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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    settings.whatsapp.send_message_url,
                    json={
                        "recipient": request.client_phone,
                        "message": remove_markdown_symbols(response_text),
                    },
                )
                response.raise_for_status()
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
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
                await client.post(
                    settings.whatsapp.send_message_url,
                    json={
                        "recipient": request.client_phone,
                        "message": "Что-то вотсап барахлит 😔. Напишите позже, пожалуйста!",
                    },
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
        logger.warning(
            f"[initConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return {"success": False, "error": "Invalid phone number"}

    request.client_phone = normalized_phone
    background_tasks.add_task(init_conversation_background, request)
    return {"success": True}


class ClientProfileResponse(BaseModel):
    """Модель ответа с профилем клиента."""

    client_phone: str
    profile: str
    message_count: int
    last_order: Optional[Dict[str, Any]] = None
    status: str


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
        from agents.tools import get_client_profile
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
        logger.warning(
            f"[resetConversation] Невалидный номер телефона: {request.client_phone}"
        )
        return {"success": False, "error": "Invalid phone number"}

    request.client_phone = normalized_phone
    background_tasks.add_task(reset_conversation_background, request)
    return {"success": True}


@router.get("/conversation-history/{phone}")
async def get_conversation_history(phone: str, days: int = 7):
    """
    Get conversation history from LangFuse for a specific phone number

    Args:
        phone: Phone number of the client
        days: Number of days to look back (default: 7)

    Returns:
        Dictionary with conversation history
    """
    normalized_phone = normalize_phone(phone)
    if not validate_phone(normalized_phone):
        logger.warning(f"[get_conversation_history] Невалидный номер телефона: {phone}")
        return {
            "phone": phone,
            "error": "Invalid phone number",
            "total_conversations": 0,
            "history": [],
        }

    try:
        from langfuse import Langfuse

        langfuse = Langfuse(
            public_key=settings.langfuse.langfuse_public_key,
            secret_key=settings.langfuse.langfuse_secret_key,
            host=settings.langfuse.langfuse_host,
        )

        from_timestamp = datetime.now() - timedelta(days=days)

        history = []
        try:
            if hasattr(langfuse, "client") and hasattr(langfuse.client, "traces"):
                try:
                    response = langfuse.client.traces.list(
                        user_id=normalized_phone,
                        from_timestamp=(
                            from_timestamp.isoformat() if from_timestamp else None
                        ),
                        limit=100,
                    )

                    if hasattr(response, "data") and response.data:
                        for trace in response.data:
                            trace_dict = (
                                trace if isinstance(trace, dict) else trace.__dict__
                            )
                            history.append(
                                {
                                    "trace_id": trace_dict.get("id"),
                                    "timestamp": trace_dict.get("timestamp"),
                                    "input": trace_dict.get("input", {}),
                                    "output": trace_dict.get("output", {}),
                                    "metadata": trace_dict.get("metadata", {}),
                                    "tools_used": trace_dict.get("metadata", {}).get(
                                        "tools_used", []
                                    ),
                                }
                            )
                except AttributeError:
                    logger.warning(f"LangFuse API structure differs from expected")
            else:
                logger.warning(f"LangFuse client does not have expected API structure")

        except Exception as api_error:
            logger.warning(
                f"Failed to fetch traces using LangFuse API: {api_error}. "
                f"Please check LangFuse dashboard directly for user_id: {normalized_phone}"
            )
            return {
                "phone": normalized_phone,
                "error": f"Could not fetch traces from API: {str(api_error)}",
                "message": f"Please check LangFuse dashboard for user_id: {normalized_phone}",
                "total_conversations": 0,
                "history": [],
            }


        return {
            "phone": normalized_phone,
            "total_conversations": len(history),
            "days": days,
            "history": history,
        }

    except Exception as e:
        logger.error(
            f"[get_conversation_history] Ошибка получения истории для {normalized_phone}: {e}",
            exc_info=True,
        )
        return {
            "phone": normalized_phone,
            "error": str(e),
            "total_conversations": 0,
            "history": [],
        }
