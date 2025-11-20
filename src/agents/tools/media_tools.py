"""Инструменты для работы с медиа (фотографии товаров)."""

from __future__ import annotations

import logging
from typing import List

import httpx
from langchain_core.tools import tool

from src.config.constants import HTTP_TIMEOUT_SECONDS
from src.config.settings import settings
from src.utils import get_supabase_client

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
            logger.debug(
                f"[send_whatsapp_image] Файл успешно отправлен для {phone}, "
                f"статус: {response.status_code}"
            )
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


def create_media_tools(client_phone: str, is_init_message: bool = False):
    @tool
    async def show_product_photos(product_ids: List[int]) -> str:
        """Отправляет фотографии товаров клиенту через WhatsApp.

        НАЗНАЧЕНИЕ: Отправляет фотографии товаров клиенту через WhatsApp

        Args:
            product_ids: Список ID товаров для отправки фото.
            Извлеки из секции [PRODUCT_IDS] ответа инструментов поиска.

        Returns:
            Статус отправки фотографий (количество отправленных, не отправленных, не найденных товаров)
        """
        logger.info(f"[show_product_photos] Sending photos to {client_phone} for product_ids: {product_ids}")

        if not product_ids:
            return "Нет товаров для отправки фотографий."

        MAX_PHOTOS_NORMAL = 3
        MAX_PHOTOS_INIT = 2
        
        original_count = len(product_ids)
        
        if is_init_message:
            product_ids = product_ids[:MAX_PHOTOS_INIT]
            if original_count > MAX_PHOTOS_INIT:
                logger.info(f"[show_product_photos] Init conversation: ограничено с {original_count} до {MAX_PHOTOS_INIT} товаров")
            else:
                logger.info(f"[show_product_photos] Init conversation: отправка {len(product_ids)} фото")
        else:
            product_ids = product_ids[:MAX_PHOTOS_NORMAL]
            if original_count > MAX_PHOTOS_NORMAL:
                logger.warning(f"[show_product_photos] Обычный запрос: получено {original_count} ID, ограничено до {MAX_PHOTOS_NORMAL} фото")
            else:
                logger.info(f"[show_product_photos] Обычный запрос: отправка {len(product_ids)} фото")

        has_photo = []
        no_photo = []
        not_found = []

        supabase = await get_supabase_client()

        for product_id in product_ids:
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
                f"УСПЕШНО ОТПРАВЛЕНО: Фотографии {len(has_photo)} товаров успешно отправлены клиенту через WhatsApp. "
                f"Клиент получил эти фотографии."
            )
        if no_photo:
            result_parts.append(
                f"НЕ ОТПРАВЛЕНО: Не удалось отправить фотографии {len(no_photo)} товаров. "
                f"Товары найдены в базе данных, но фотографии либо отсутствуют, либо произошла ошибка при отправке. "
                f"ВАЖНО: Несмотря на ошибку отправки фото, ты ДОЛЖЕН продолжить и сформировать текстовое сообщение с информацией о товарах для клиента."
            )
        if not_found:
            result_parts.append(
                f"НЕ НАЙДЕНО: {len(not_found)} товаров не найдены в базе данных. "
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
