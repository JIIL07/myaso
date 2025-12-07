"""Инструменты для работы с медиа (фотографии товаров)."""

from __future__ import annotations

import logging

import httpx
from langchain_core.tools import tool

from src.utils.rules import get_rule_as_float
from src.config.settings import settings
from src.utils.supabase_client import get_supabase_client
from src.agents.tools.context_vars import get_client_phone
from src.agents.tools.context_tools import get_product_ids_from_context

logger = logging.getLogger(__name__)

async def send_whatsapp_image(phone: str, file_url: str, caption: str, extension: str = "png") -> bool:
    """Отправляет файл через WhatsApp API.

    Args:
        phone: Номер телефона получателя
        file_url: URL файла
        caption: Подпись к файлу
        extension: Тип файла (по умолчанию "png")

    Returns:
        True если файл успешно отправлен, False в случае ошибки
    """
    try:
        timeout = await get_rule_as_float("HTTP_TIMEOUT_SECONDS")
    except Exception:
        timeout = 10.0
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                url=settings.whatsapp.send_file_url,
                json={
                    "recipient": phone,
                    "file_url": file_url,
                    "caption": caption,
                    "extension": extension,
                },
            )
            response.raise_for_status()
            return True
    except httpx.HTTPStatusError as e:
        logger.error(f"[send_whatsapp_image] HTTP ошибка для {phone}: статус {e.response.status_code}, файл: {file_url}")
        return False
    except Exception as e:
        logger.error(f"[send_whatsapp_image] Ошибка для {phone}: {e}, файл: {file_url}")
        return False


@tool
async def show_product_photos() -> str:
    """Отправляет фотографии товаров клиенту через WhatsApp.

    КРИТИЧЕСКИ ВАЖНО: 
    - Инструмент НЕ принимает параметры - вызывай БЕЗ аргументов: show_product_photos()
    - ID товаров автоматически берутся из контекста агента
    - Контекст автоматически заполняется после вызова инструментов поиска товаров:
      * vector_search
      * execute_sql_query
      * get_random_products
      * find_similar_products
    - НЕ передавай ID вручную - они уже сохранены в контексте через artifacts

    НАЗНАЧЕНИЕ: Отправляет фотографии товаров клиенту через WhatsApp

    Returns:
        Статус отправки фотографий (количество отправленных, не отправленных, не найденных товаров)
    """
    client_phone = get_client_phone()
    parsed_ids = get_product_ids_from_context(client_phone)
    
    if not parsed_ids:
        return "Нет сохраненных ID товаров для отправки фотографий. Используйте инструменты поиска товаров (vector_search, execute_sql_query, get_random_products) для поиска товаров."    
    has_photo = []
    no_photo = []
    not_found = []

    supabase = await get_supabase_client()

    for product_id in parsed_ids:
        try:
            result = (
                await supabase.table("products")
                .select("*")
                .eq("id", product_id)
                .execute()
            )

            if not result.data:
                not_found.append(product_id)
                continue

            product = result.data[0]
            photo_url = product.get("photo")
            product_title = product.get("title", f"Товар #{product_id}")

            if photo_url and await send_whatsapp_image(client_phone, photo_url, product_title):
                has_photo.append(product_id)
            else:
                no_photo.append(product_id)

        except Exception as e:
            logger.error(
                f"[show_product_photos] Ошибка при получении товара ID {product_id}: {e}",
                exc_info=True
            )
            not_found.append(product_id)

    result_parts = []
    if has_photo:
        result_parts.append(f"✅ УСПЕШНО ОТПРАВЛЕНО: Фотографии {len(has_photo)} товаров успешно отправлены клиенту через WhatsApp. Клиент получил эти фотографии.")
    if no_photo:
        result_parts.append(
            f"❌ НЕ ОТПРАВЛЕНО: Не удалось отправить фотографии {len(no_photo)} товаров. "
            f"Товары найдены в базе данных, но фотографии либо отсутствуют, либо произошла ошибка при отправке. "
            f"\n\nКРИТИЧЕСКИ ВАЖНО: Несмотря на ошибку отправки фото, ты ОБЯЗАТЕЛЬНО ДОЛЖЕН:\n"
            f"1. НЕ говорить что фото отправлены\n"
            f"2. ВСЕГДА предложить товары текстом с полной информацией (название, поставщик, цена, регион)\n"
            f"3. Сообщить что фото временно недоступны, но товары есть в наличии"
        )
    if not_found:
        result_parts.append(f"⚠️ НЕ НАЙДЕНО: {len(not_found)} товаров не найдены в базе данных. Эти товары отсутствуют в каталоге.")
    
    return "\n".join(result_parts) if result_parts else "Нет товаров для отправки фотографий."


@tool
async def send_pricelist() -> str:
    """Отправляет прайс-лист клиенту через WhatsApp.
    
    Используй этот инструмент когда:
    - Клиент просит прайс-лист
    - Клиент спрашивает про каталог товаров
    - Нужно отправить полный список товаров с ценами
    
    Returns:
        Статус отправки прайс-листа
    """
    from src.utils.prompts import get_system_value
    from src.config.messages_constants import SYSTEM_VALUE_PRICELIST
    from urllib.parse import urlparse
    import os
    
    client_phone = get_client_phone()
    
    try:
        pricelist_url = await get_system_value(SYSTEM_VALUE_PRICELIST)
        if not pricelist_url:
            return "Прайс-лист не настроен в системе. Сообщи клиенту, что прайс-лист временно недоступен."
        
        file_extension = os.path.splitext(urlparse(pricelist_url).path)[1].lstrip('.').lower() or "xlsx"
        
        send_success = await send_whatsapp_image(
            phone=client_phone,
            file_url=pricelist_url,
            caption=SYSTEM_VALUE_PRICELIST,
            extension=file_extension,
        )
        
        if send_success:
            return "✅ Прайс-лист успешно отправлен клиенту через WhatsApp. Клиент получил файл с прайс-листом."
        else:
            return (
                "❌ Не удалось отправить прайс-лист. "
                "Сообщи клиенту, что прайс-лист временно недоступен, но ты можешь помочь с поиском товаров."
            )
    except Exception as e:
        logger.error(f"[send_pricelist] Ошибка: {e}", exc_info=True)
        return (
            "❌ Ошибка при отправке прайс-листа. "
            "Сообщи клиенту, что прайс-лист временно недоступен, но ты можешь помочь с поиском товаров."
        )
