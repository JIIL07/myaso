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
    DEFAULT_SQL_LIMIT,
    MAX_SQL_RETRY_ATTEMPTS,
    HTTP_TIMEOUT_SECONDS,
)
from agents.factory import AgentFactory
from src.utils import remove_markdown_symbols, extract_product_titles_from_text
from src.utils.langchain_memory import SupabaseConversationMemory
from src.utils.phone_validator import normalize_phone, validate_phone
from agents.tools import (
    get_client_profile,
    get_random_products,
    generate_sql_from_text,
    execute_sql_conditions,
    show_product_photos,
)
from supabase import acreate_client, AClient, AsyncClientOptions
from src.config.settings import settings
from src.utils.prompts import get_prompt
import httpx

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai")


async def process_conversation_background(request: UserMessageRequest):
    """Обрабатывает запрос пользователя в фоновом режиме.

    Args:
        request: Запрос с сообщением пользователя и номером телефона
    """
    logger.info(
        f"[processConversation] Начало обработки запроса для {request.client_phone}"
    )

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

        logger.info(
            f"[processConversation] Получен ответ от агента для {request.client_phone}, длина: {len(response_text)}"
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
            logger.info(
                f"[processConversation] Сообщение отправлено в WhatsApp для {request.client_phone}"
            )
        except Exception as e:
            logger.warning(f"[processConversation] Ошибка отправки в WhatsApp: {e}")

        logger.info(
            f"[processConversation] Завершение обработки для {request.client_phone}"
        )
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
    logger.info(f"[processConversation] Получен запрос от {request.client_phone}")
    background_tasks.add_task(process_conversation_background, request)
    return {"success": True}


async def init_conversation_background(request: InitConverastionRequest):
    """Инициализирует новую беседу с клиентом в фоновом режиме.

    Args:
        request: Запрос с номером телефона клиента и темой беседы
    """
    logger.info(
        f"[initConversation] Начало обработки запроса для {request.client_phone}, topic: {request.topic}"
    )

    try:
        memory = await SupabaseConversationMemory(request.client_phone)
        await memory.clear()

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        profile_text = (
            await get_client_profile.ainvoke({"phone": request.client_phone})
            or "Профиль клиента не найден."
        )

        products_text = ""
        product_titles = []
        text_conditions = request.topic
        sql_conditions = None
        last_error = None
        sql_success = False

        for attempt in range(1, MAX_SQL_RETRY_ATTEMPTS + 1):
            try:
                invoke_params = {
                    "text_conditions": text_conditions,
                    "attempt_number": attempt,
                    "topic": request.topic,
                }

                if attempt > 1 and sql_conditions:
                    invoke_params["previous_sql"] = sql_conditions

                if attempt > 1 and last_error:
                    invoke_params["error_message"] = str(last_error)

                sql_conditions = await generate_sql_from_text.ainvoke(invoke_params)
                products_text = await execute_sql_conditions.ainvoke(
                    {"sql_conditions": sql_conditions, "limit": DEFAULT_SQL_LIMIT}
                )
                if products_text and "не найдены" not in products_text.lower():
                    sql_success = True
                    product_titles = extract_product_titles_from_text(products_text)
                    break
                else:
                    raise ValueError("Товары не найдены")
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[initConversation] Попытка {attempt} SQL запроса не удалась: {e}"
                )

        if not sql_success:
            logger.warning(
                f"[initConversation] Все {MAX_SQL_RETRY_ATTEMPTS} попытки SQL запроса не удались, используем случайные товары"
            )
            try:
                products_text = await get_random_products.ainvoke({"limit": 10})
                product_titles = extract_product_titles_from_text(products_text)
            except Exception:
                products_text = "Ассортимент будет обновлён позже."
                product_titles = []

        context_parts = []
        context_parts.append(
            "Сформируй короткое дружелюбное приветствие для клиента, учитывая его профиль и ассортимент.\n"
        )
        context_parts.append(f"Тема диалога: {request.topic}\n\n")
        context_parts.append(f"Профиль клиента:\n{profile_text}\n\n")
        context_parts.append(f"Ассортимент/подборка:\n{products_text}\n\n")
        context_parts.append(
            "Поприветствуй дружелюбно со смайликами, будь позитивным и энергичным. Предложи помощь и ненавязчиво уточни запрос."
        )

        welcome_input = "".join(context_parts)
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
                logger.info(
                    f"[initConversation] Сообщение успешно отправлено для {request.client_phone}"
                )
        except Exception as send_error:
            logger.error(
                f"[initConversation] Ошибка отправки сообщения в WhatsApp для {request.client_phone}: {send_error}",
                exc_info=True,
            )
            raise

        if product_titles:
            logger.info(
                f"[initConversation] Отправка фотографий {len(product_titles)} товаров для {request.client_phone}"
            )
            try:
                photos_result = await show_product_photos.ainvoke(
                    {"product_titles": product_titles, "phone": request.client_phone}
                )
                logger.info(
                    f"[initConversation] Результат отправки фото: {photos_result}"
                )
            except Exception as photo_error:
                logger.warning(
                    f"[initConversation] Ошибка отправки фотографий: {photo_error}"
                )

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
    logger.info(
        f"[initConversation] Получен запрос от {request.client_phone}, topic: {request.topic}"
    )
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
        profile_text = await get_client_profile.ainvoke({"phone": client_phone})
    except Exception:
        profile_text = "Профиль клиента не найден в базе данных."

    message_count = 0
    last_order: Optional[Dict[str, Any]] = None
    supabase: AClient | None = None

    try:
        supabase = await acreate_client(
            settings.supabase.supabase_url,
            settings.supabase.supabase_service_key,
            options=AsyncClientOptions(schema="myaso"),
        )

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

        logger.info(
            f"[get_conversation_history] Найдено {len(history)} разговоров для {normalized_phone}"
        )

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
