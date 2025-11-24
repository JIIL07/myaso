"""Инструменты для работы с медиа (фотографии товаров)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Union

import httpx
from langchain_core.tools import tool

from src.config.constants import HTTP_TIMEOUT_SECONDS
from src.config.settings import settings
from src.utils import get_supabase_client
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
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
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
        logger.error(
            f"[send_whatsapp_image] Ошибка HTTP при отправке файла для {phone}: "
            f"статус {e.response.status_code}, файл: {file_url}, ошибка: {e}"
        )
        return False
    except Exception as e:
        logger.error(
            f"[send_whatsapp_image] Ошибка отправки файла для {phone}: {e}, "
            f"file: {file_url}"
        )
        return False


def _parse_product_ids(product_ids: Union[List[int], List[str], str]) -> List[int]:
    """Парсит ID товаров из различных форматов.
    
    Args:
        product_ids: ID товаров в формате:
            - List[int] - список чисел
            - List[str] - список строк (преобразуется в int)
            - str - строка с JSON {"product_ids": [...]} или "[PRODUCT_IDS]...[/PRODUCT_IDS]"
    
    Returns:
        List[int] - список ID товаров
    """
    if not product_ids:
        return []
    
    # Если это уже список чисел
    if isinstance(product_ids, list):
        parsed_ids = []
        for pid in product_ids:
            try:
                if isinstance(pid, int):
                    parsed_ids.append(pid)
                elif isinstance(pid, str):
                    parsed_id = int(pid.strip())
                    parsed_ids.append(parsed_id)
            except (ValueError, TypeError) as e:
                logger.warning(f"[_parse_product_ids] Не удалось преобразовать ID '{pid}' в число: {e}")
        return parsed_ids
    
    # Если это строка, пытаемся извлечь JSON
    if isinstance(product_ids, str):
        product_ids_str = product_ids.strip()
        
        # Пытаемся извлечь из [PRODUCT_IDS]...[/PRODUCT_IDS]
        match = re.search(r'\[PRODUCT_IDS\](.*?)\[/PRODUCT_IDS\]', product_ids_str, re.DOTALL)
        if match:
            product_ids_str = match.group(1).strip()
        
        # Пытаемся распарсить как JSON
        try:
            data = json.loads(product_ids_str)
            if isinstance(data, dict) and "product_ids" in data:
                ids_list = data["product_ids"]
            elif isinstance(data, list):
                ids_list = data
            else:
                logger.warning(f"[_parse_product_ids] Неожиданный формат JSON: {data}")
                return []
            
            # Преобразуем в список int
            parsed_ids = []
            for pid in ids_list:
                try:
                    parsed_id = int(pid) if not isinstance(pid, int) else pid
                    parsed_ids.append(parsed_id)
                except (ValueError, TypeError) as e:
                    logger.warning(f"[_parse_product_ids] Не удалось преобразовать ID '{pid}' в число: {e}")
            return parsed_ids
        except json.JSONDecodeError:
            # Если не JSON, пытаемся извлечь числа из строки
            logger.warning(f"[_parse_product_ids] Не удалось распарсить JSON, пытаемся извлечь числа из строки: {product_ids_str[:100]}")
            numbers = re.findall(r'\d+', product_ids_str)
            parsed_ids = []
            for num_str in numbers:
                try:
                    parsed_ids.append(int(num_str))
                except ValueError:
                    pass
            return parsed_ids
    
    return []


@tool
async def show_product_photos(product_ids: Any = None) -> str:
    """Отправляет фотографии товаров клиенту через WhatsApp.

    Если product_ids не указан, использует ID из контекста агента (agent_context).
    Если указан, использует переданные ID (для обратной совместимости).

    НАЗНАЧЕНИЕ: Отправляет фотографии товаров клиенту через WhatsApp

    Args:
        product_ids: ID товаров (опционально). Если None, берется из контекста агента.
        Может быть:
        - None - берется из контекста агента
        - Список чисел: [1, 2, 3]
        - Строка с JSON: '{"product_ids": [1, 2, 3]}'
        - Строка с [PRODUCT_IDS]: '[PRODUCT_IDS]{"product_ids": [1, 2, 3]}[/PRODUCT_IDS]'

    Returns:
        Статус отправки фотографий (количество отправленных, не отправленных, не найденных товаров)
    """
    client_phone = get_client_phone()
    logger.info(f"[show_product_photos] Получены product_ids (тип: {type(product_ids)}): {product_ids}")
    
    # Если product_ids не указан, берем из контекста
    if product_ids is None:
        parsed_ids = get_product_ids_from_context(client_phone)
        if not parsed_ids:
            return "Нет сохраненных ID товаров для отправки фотографий. Используйте инструменты поиска товаров (vector_search, execute_sql_query, get_random_products) для поиска товаров."
        logger.info(f"[show_product_photos] Используются product_ids из контекста: {parsed_ids}")
    else:
        # Парсим ID товаров из различных форматов (для обратной совместимости)
        parsed_ids = _parse_product_ids(product_ids)
    
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

            if not result.data or len(result.data) == 0:
                not_found.append(product_id)
                logger.warning(f"[show_product_photos] Товар с ID {product_id} не найден в базе данных")
                continue

            product = result.data[0]
            photo_url = product.get("photo")
            product_title = product.get("title", f"Товар #{product_id}")

            if photo_url:
                send_success = await send_whatsapp_image(client_phone, photo_url, product_title)
                if send_success:
                    has_photo.append(product_id)
                    logger.info(
                        f"[show_product_photos] Фото успешно отправлено для товара ID {product_id} "
                        f"('{product_title}') на номер {client_phone}"
                    )
                else:
                    no_photo.append(product_id)
                    logger.warning(
                        f"[show_product_photos] Не удалось отправить фото для товара ID {product_id} "
                        f"('{product_title}') на номер {client_phone}"
                    )
            else:
                no_photo.append(product_id)
                logger.info(f"[show_product_photos] Товар ID {product_id} ('{product_title}') найден, но нет фотографии")

        except Exception as e:
            logger.error(
                f"[show_product_photos] Ошибка при получении товара ID {product_id}: {e}",
                exc_info=True
            )
            not_found.append(product_id)

    result_parts = []
    
    if has_photo:
        result_parts.append(
            f"✅ УСПЕШНО ОТПРАВЛЕНО: Фотографии {len(has_photo)} товаров успешно отправлены клиенту через WhatsApp. "
            f"Клиент получил эти фотографии."
        )
    
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
        result_parts.append(
            f"⚠️ НЕ НАЙДЕНО: {len(not_found)} товаров не найдены в базе данных. "
            f"Эти товары отсутствуют в каталоге."
        )

    result_text = (
        "\n".join(result_parts)
        if result_parts
        else "Нет товаров для отправки фотографий."
    )

    logger.info(
        f"[show_product_photos] Итоговый результат для {client_phone}: "
        f"отправлено={len(has_photo)} (IDs: {has_photo}), "
        f"не отправлено={len(no_photo)} (IDs: {no_photo}), "
        f"не найдено={len(not_found)} (IDs: {not_found})"
    )

    return result_text


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
        
        parsed_url = urlparse(pricelist_url)
        file_path = parsed_url.path
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension:
            file_extension = file_extension.lstrip('.').lower()
        else:
            file_extension = "xlsx"
        
        send_success = await send_whatsapp_image(
            phone=client_phone,
            file_url=pricelist_url,
            caption=SYSTEM_VALUE_PRICELIST,
            extension=file_extension,
        )
        
        if send_success:
            logger.info(f"[send_pricelist] Прайс-лист успешно отправлен для {client_phone}")
            return "✅ Прайс-лист успешно отправлен клиенту через WhatsApp. Клиент получил файл с прайс-листом."
        else:
            logger.warning(f"[send_pricelist] Не удалось отправить прайс-лист для {client_phone}")
            return (
                "❌ Не удалось отправить прайс-лист. "
                "Сообщи клиенту, что прайс-лист временно недоступен, но ты можешь помочь с поиском товаров."
            )
    except Exception as e:
        logger.error(f"[send_pricelist] Ошибка отправки прайс-листа для {client_phone}: {e}", exc_info=True)
        return (
            "❌ Ошибка при отправке прайс-листа. "
            "Сообщи клиенту, что прайс-лист временно недоступен, но ты можешь помочь с поиском товаров."
        )
