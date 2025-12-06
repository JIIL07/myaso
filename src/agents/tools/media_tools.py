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
from src.agents.tools.context_tools import get_is_init_message

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


def create_media_tools(client_phone: str, is_init_message: bool = False):
    """Создает инструменты для работы с медиа.

    Args:
        client_phone: Номер телефона клиента
        is_init_message: Если True, это init conversation (используется для обратной совместимости)

    Returns:
        Список инструментов для работы с медиа
    """
    @tool
    async def show_product_photos(product_ids: Any) -> str:
        """Отправляет фотографии товаров клиенту через WhatsApp.

        НАЗНАЧЕНИЕ: Отправляет фотографии товаров клиенту через WhatsApp

        Args:
            product_ids: ID товаров для отправки фото.
            Может быть:
            - Список чисел: [1, 2, 3]
            - Строка с JSON: '{"product_ids": [1, 2, 3]}'
            - Строка с [PRODUCT_IDS]: '[PRODUCT_IDS]{"product_ids": [1, 2, 3]}[/PRODUCT_IDS]'
            Извлеки ID из секции [PRODUCT_IDS] ответа инструментов поиска (vector_search, execute_sql_query, get_random_products).

        Returns:
            Статус отправки фотографий (количество отправленных, не отправленных, не найденных товаров)
        """
        logger.info(f"[show_product_photos] Получены product_ids (тип: {type(product_ids)}): {product_ids}")
        
        # Парсим ID товаров из различных форматов
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

    return [show_product_photos]
