import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks
from supabase import AClient

from src.agents.factory import AgentFactory
from src.config.settings import settings
from src.models import (
    ClientProfileResponse,
    InitConverastionRequest,
    ResetConversationRequest,
    UserMessageRequest,
)
from src.services.whatsapp_service import send_image, send_message
from src.utils import get_supabase_client, remove_markdown_symbols
from src.utils.memory import SupabaseConversationMemory
from src.utils.phone_validator import normalize_phone, validate_phone
from src.utils.prompts import get_system_value

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai")


async def process_conversation_background(request: UserMessageRequest):
    """Обрабатывает запрос пользователя в фоновом режиме.

    Args:
        request: Запрос с сообщением пользователя и номером телефона
    """

    try:
        logger.info(
            f"[processConversation] Получен запрос для {request.client_phone}: "
            f"message='{request.message}', topic='{request.topic}'"
        )
        
        memory = await SupabaseConversationMemory(request.client_phone)
        logger.info(f"[processConversation] Память создана для {request.client_phone}, async_initialized={getattr(memory, 'async_initialized', False)}")

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
        
        if not hasattr(memory, 'async_initialized') or not memory.async_initialized:
            logger.warning(f"[initConversation] Память не инициализирована для {request.client_phone}, инициализируем...")
            await memory.__ainit__(request.client_phone)
        
        await memory.clear()

        factory = AgentFactory.instance()
        agent = factory.create_product_agent(config={"memory": memory})

        welcome_input = f"""Сформируй короткое дружелюбное приветствие для клиента.

ВАЖНО: Это init_conversation - инициализация разговора. Ты ДОЛЖЕН выполнить ВСЕ шаги ниже:

ШАГ 1: Получи профиль клиента
- Вызови get_client_profile(phone="{request.client_phone}")
- Проверь статус дружбы клиента (it_is_friend) в профиле
- Если it_is_friend=TRUE - обращайся на "ты", если FALSE - на "вы"

ШАГ 2: Найди товары
- Используй generate_sql_from_text + execute_sql_query для поиска товаров
- При вызове execute_sql_query укажи limit
- Для каждого товара рассчитай финальную цену по правилам из промпта

ШАГ 3: Отправь фото товаров (если есть)
- После того как execute_sql_query вернул товары, найди в ответе инструмента секцию [PRODUCT_IDS]
- Формат секции: [PRODUCT_IDS]{{"product_ids": [123, 456]}}[/PRODUCT_IDS]
- Извлеки числа из массива product_ids (это ID товаров)
- Возьми товары по limit ID из этого массива (например, если [123, 456, 789], то возьми [123, 456])
- ВЫЗОВИ инструмент show_product_photos с этими ID:
  show_product_photos(product_ids=[123, 456])
- Инструмент show_product_photos автоматически отправит фото только для тех товаров, у которых есть фотографии
- Если у товара нет фото, оно не будет отправлено, но товар всё равно будет показан в текстовом сообщении

ШАГ 4: Сформируй приветственное сообщение
- Дружелюбное приветствие с учетом статуса дружбы (ты/вы)
- Краткое введение: "Предлагаю вам актуальные позиции из нашего ассортимента:"
- Для каждого из найденных товаров укажи:
  * Название товара (title) - ТОЧНО как в базе данных
  * Поставщик (supplier_name): "Поставщик: {{supplier_name}}"
  * Регион происхождения (from_region): "Регион: {{from_region}}"
  * Финальная цена за килограм (РАССЧИТАННАЯ): "Цена: {{final_price_kg:.2f}} р/кг" или "Цена: по запросу" если цена = 0

ПРИМЕР ПОЛНОГО ЦИКЛА:
1. get_client_profile(phone="{request.client_phone}")
2. generate_sql_from_text(text_conditions="найди товары")
3. execute_sql_query(sql_query="...")
4. show_product_photos(product_ids=[123, 456])  # Отправит фото только если они есть у товаров
5. Сформируй приветственное сообщение с информацией о товарах

Поприветствуй дружелюбно со смайликами, будь позитивным и энергичным.
"""

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

        # Отправка прайс-листа после текста и фото
        try:
            pricelist_url = await get_system_value("Прайс-лист")
            if pricelist_url:
                logger.info(
                    f"[initConversation] Найден прайс-лист для {request.client_phone}: {pricelist_url}"
                )
                
                # Определяем расширение файла из URL
                parsed_url = urlparse(pricelist_url)
                file_path = parsed_url.path
                _, file_extension = os.path.splitext(file_path)
                
                # Убираем точку из расширения, если есть
                if file_extension:
                    file_extension = file_extension.lstrip('.')
                else:
                    # Если расширение не найдено, пытаемся определить по параметрам URL или используем pdf по умолчанию
                    file_extension = "xlsx"
                
                # Приводим расширение к нижнему регистру для совместимости
                file_extension = file_extension.lower()
                
                # Отправляем прайс-лист как файл
                send_file_success = await send_image(
                    recipient=request.client_phone,
                    file_url=pricelist_url,
                    caption="Прайс-лист",
                    extension=file_extension,
                )
                
                if send_file_success:
                    logger.info(
                        f"[initConversation] Прайс-лист успешно отправлен для {request.client_phone}"
                    )
                else:
                    logger.warning(
                        f"[initConversation] Не удалось отправить прайс-лист для {request.client_phone}"
                    )
            else:
                logger.info(
                    f"[initConversation] Прайс-лист не найден в system table для {request.client_phone}"
                )
        except Exception as pricelist_error:
            logger.error(
                f"[initConversation] Ошибка при отправке прайс-листа для {request.client_phone}: {pricelist_error}",
                exc_info=True,
            )
            # Не прерываем выполнение, так как основное сообщение уже отправлено

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