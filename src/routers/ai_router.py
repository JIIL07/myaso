from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from src.schemas import UserMessageRequest, InitConverastionRequest, ResetConversationRequest
from agents.factory import AgentFactory
from src.utils import remove_markdown_symbols
from src.utils.langchain_memory import SupabaseConversationMemory
from src.utils.phone_validator import normalize_phone, validate_phone
from agents.tools import (
    get_client_profile,
    get_random_products,
    generate_sql_from_text,
    execute_sql_conditions,
    show_product_photos,
    get_product_titles_from_sql_conditions,
    get_random_product_titles,
)
from supabase import acreate_client, AClient, AsyncClientOptions
from src.config.settings import settings
from src.utils.prompts import get_prompt
import httpx
from langfuse import Langfuse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai")


async def process_conversation_background(request: UserMessageRequest):
    """Обрабатывает запрос пользователя в фоновом режиме.
    
    Args:
        request: Запрос с сообщением пользователя и номером телефона
    """
    logger.info(f"[processConversation] Начало обработки запроса для {request.client_phone}")
    langfuse = None
    trace = None

    try:
        try:
            langfuse = Langfuse()
            trace = langfuse.trace(
                name="processConversation",
                user_id=request.client_phone,
                input={
                    "client_phone": request.client_phone,
                    "message": request.message,
                },
                tags=["langchain", "agent"],
            )
        except Exception as _:
            langfuse = None
            trace = None

        memory = await SupabaseConversationMemory(request.client_phone)
        
        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        response_text = await agent.run(
            user_input=request.message, 
            client_phone=request.client_phone,
            topic=request.topic
        )
        
        logger.info(f"[processConversation] Получен ответ от агента для {request.client_phone}, длина: {len(response_text)}")

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    settings.whatsapp.send_message_url,
                    json={
                        "recipient": request.client_phone,
                        "message": remove_markdown_symbols(response_text),
                    },
                )
            logger.info(f"[processConversation] Сообщение отправлено в WhatsApp для {request.client_phone}")
        except Exception as e:
            logger.warning(f"[processConversation] Ошибка отправки в WhatsApp: {e}")

        if trace is not None:
            try:
                trace.update(output={"response": response_text})
            except Exception:
                pass

        logger.info(f"[processConversation] Завершение обработки для {request.client_phone}")
        return {"success": True}

    except Exception as e:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    settings.whatsapp.send_message_url,
                    json={
                        "recipient": request.client_phone,
                        "message": "Что-то вотсап барахлит 😔. Напишите позже, пожалуйста!",
                    },
                )
        except Exception:
            pass

        if trace is not None:
            try:
                trace.update(output={"error": str(e)})
            except Exception:
                pass

        return {"success": False}


@router.post("/processConversation", status_code=200)
async def process_conversation(request: UserMessageRequest, background_tasks: BackgroundTasks):
    """Обрабатывает запрос пользователя и запускает фоновую задачу.
    
    Args:
        request: Запрос с сообщением пользователя
        background_tasks: Фоновые задачи FastAPI
        
    Returns:
        Словарь с результатом успешного запуска задачи
    """
    normalized_phone = normalize_phone(request.client_phone)
    if not validate_phone(normalized_phone):
        logger.warning(f"[processConversation] Невалидный номер телефона: {request.client_phone}")
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
    logger.info(f"[initConversation] Начало обработки запроса для {request.client_phone}, topic: {request.topic}")
    langfuse = None
    trace = None
    try:
        langfuse = Langfuse()
        trace = langfuse.trace(
            name="initConversation",
            user_id=request.client_phone,
            input={"client_phone": request.client_phone, "topic": request.topic},
            tags=["langchain", "agent", "init"],
        )
    except Exception:
        pass

    try:
        memory = await SupabaseConversationMemory(request.client_phone)
        await memory.clear()

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        profile_text = await get_client_profile.ainvoke({"phone": request.client_phone}) or "Профиль клиента не найден."

        products_text = ""
        product_titles = []
        text_conditions = request.topic
        sql_conditions = None
        last_error = None
        sql_success = False
        
        for attempt in range(1, 4):
            try:
                sql_conditions = await generate_sql_from_text.ainvoke({
                    "text_conditions": text_conditions,
                    "previous_sql": sql_conditions if attempt > 1 else None,
                    "error_message": str(last_error) if attempt > 1 and last_error else None,
                    "attempt_number": attempt
                })
                products_text = await execute_sql_conditions.ainvoke({"sql_conditions": sql_conditions, "limit": 15})
                if products_text and "не найдены" not in products_text.lower():
                    sql_success = True
                    product_titles = await get_product_titles_from_sql_conditions.ainvoke({
                        "sql_conditions": sql_conditions,
                        "limit": 15
                    })
                    break
                else:
                    raise ValueError("Товары не найдены")
            except Exception as e:
                last_error = e
                logger.warning(f"[initConversation] Попытка {attempt} SQL запроса не удалась: {e}")
        
        if not sql_success:
            logger.warning(f"[initConversation] Все 3 попытки SQL запроса не удались, используем случайные товары")
            try:
                products_text = await get_random_products.ainvoke({"limit": 10})
                product_titles = await get_random_product_titles.ainvoke({"limit": 10})
            except Exception:
                products_text = "Ассортимент будет обновлён позже."

        context_parts = []
        context_parts.append("Сформируй короткое дружелюбное приветствие для клиента, учитывая его профиль и ассортимент.\n")
        context_parts.append(f"Тема диалога: {request.topic}\n\n")
        context_parts.append(f"Профиль клиента:\n{profile_text}\n\n")
        context_parts.append(f"Ассортимент/подборка:\n{products_text}\n\n")
        context_parts.append("Поприветствуй дружелюбно со смайликами, будь позитивным и энергичным. Предложи помощь и ненавязчиво уточни запрос.")

        welcome_input = "".join(context_parts)
        response_text = await agent.run(user_input=welcome_input, client_phone=request.client_phone, topic=request.topic, is_init_message=True)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    settings.whatsapp.send_message_url,
                    json={"recipient": request.client_phone, "message": remove_markdown_symbols(response_text)},
                )
                response.raise_for_status()
                logger.info(f"[initConversation] Сообщение успешно отправлено для {request.client_phone}")
        except Exception as send_error:
            logger.error(f"[initConversation] Ошибка отправки сообщения в WhatsApp для {request.client_phone}: {send_error}", exc_info=True)
            raise

        if product_titles:
            logger.info(f"[initConversation] Отправка фотографий {len(product_titles)} товаров для {request.client_phone}")
            try:
                photos_result = await show_product_photos.ainvoke({
                    "product_titles": product_titles,
                    "phone": request.client_phone
                })
                logger.info(f"[initConversation] Результат отправки фото: {photos_result}")
            except Exception as photo_error:
                logger.warning(f"[initConversation] Ошибка отправки фотографий: {photo_error}")

        if trace:
            try:
                trace.update(output={"response": response_text, "products": products_text})
            except Exception:
                pass

        return {"success": True}

    except Exception as e:
        logger.error(f"[initConversation] Критическая ошибка для {request.client_phone}: {e}", exc_info=True)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    settings.whatsapp.send_message_url,
                    json={
                        "recipient": request.client_phone,
                        "message": "Что-то вотсап барахлит 😔. Напишите позже, пожалуйста!",
                    },
                )
        except Exception as send_error:
            logger.error(f"[initConversation] Ошибка отправки сообщения об ошибке: {send_error}")

        if trace is not None:
            try:
                trace.update(output={"error": str(e)})
            except Exception:
                pass

        return {"success": False}


@router.post("/initConversation", status_code=200)
async def init_conversation(request: InitConverastionRequest, background_tasks: BackgroundTasks):
    """Инициализирует новую беседу и запускает фоновую задачу.
    
    Args:
        request: Запрос с номером телефона и темой беседы
        background_tasks: Фоновые задачи FastAPI
        
    Returns:
        Словарь с результатом успешного запуска задачи
    """
    normalized_phone = normalize_phone(request.client_phone)
    if not validate_phone(normalized_phone):
        logger.warning(f"[initConversation] Невалидный номер телефона: {request.client_phone}")
        return {"success": False, "error": "Invalid phone number"}
    
    request.client_phone = normalized_phone
    logger.info(f"[initConversation] Получен запрос от {request.client_phone}, topic: {request.topic}")
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
            options=AsyncClientOptions(schema="myaso")
        )
        
        history_resp = await supabase.table('conversation_history').select('*').eq('client_phone', client_phone).execute()
        message_count = len(history_resp.data) if history_resp.data else 0
        
        orders_resp = await supabase.table('orders').select('*').eq('client_phone', client_phone).order('created_at', desc=True).execute()
        orders = orders_resp.data if orders_resp.data else []
        if orders:
            o = orders[0]  # Уже отсортировано по дате
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
    langfuse = None
    trace = None
    
    try:
        try:
            langfuse = Langfuse()
            trace = langfuse.trace(
                name="resetConversation",
                user_id=request.client_phone,
                input={"client_phone": request.client_phone},
                tags=["langchain", "agent", "reset"],
            )
        except Exception:
            langfuse = None
            trace = None

        memory = await SupabaseConversationMemory(request.client_phone)
        await memory.clear()

        if trace is not None:
            try:
                trace.update(output={"success": True})
            except Exception:
                pass

        return {"success": True}

    except Exception as e:
        if trace is not None:
            try:
                trace.update(output={"error": str(e)})
            except Exception:
                pass
        return {"success": False}


@router.delete("/resetConversation", status_code=200)
async def reset_conversation(request: ResetConversationRequest, background_tasks: BackgroundTasks):
    """Сбрасывает историю беседы и запускает фоновую задачу.
    
    Args:
        request: Запрос с номером телефона клиента
        background_tasks: Фоновые задачи FastAPI
        
    Returns:
        Словарь с результатом успешного запуска задачи
    """
    normalized_phone = normalize_phone(request.client_phone)
    if not validate_phone(normalized_phone):
        logger.warning(f"[resetConversation] Невалидный номер телефона: {request.client_phone}")
        return {"success": False, "error": "Invalid phone number"}
    
    request.client_phone = normalized_phone
    background_tasks.add_task(reset_conversation_background, request)
    return {"success": True}
