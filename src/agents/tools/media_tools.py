"""Инструменты для работы с медиа (фотографии товаров)."""

from __future__ import annotations

from typing import List
import logging
import httpx
from langchain_core.tools import tool

from src.config.settings import settings
from src.config.constants import HTTP_TIMEOUT_SECONDS
from src.utils import get_supabase_client

logger = logging.getLogger(__name__)

async def send_whatsapp_image(phone: str, image_url: str, caption: str) -> bool:
    """Отправляет изображение через WhatsApp API.

    Args:
        phone: Номер телефона получателя
        image_url: URL изображения
        caption: Подпись к изображению

    Returns:
        True если изображение успешно отправлено, False в случае ошибки
    """
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url=settings.whatsapp.send_image_url,
                json={
                    "recipient": phone,
                    "image_url": image_url,
                    "caption": caption,
                },
            )
            response.raise_for_status()
            logger.debug(
                f"[send_whatsapp_image] Фото успешно отправлено для {phone}, "
                f"статус: {response.status_code}"
            )
            return True
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[send_whatsapp_image] Ошибка HTTP при отправке фото для {phone}: "
            f"статус {e.response.status_code}, фото: {image_url}, ошибка: {e}"
        )
        return False
    except Exception as e:
        logger.error(
            f"[send_whatsapp_image] Ошибка отправки фото для {phone}: {e}, "
            f"photo: {image_url}"
        )
        return False


def create_media_tools(client_phone: str, is_init_message: bool = False):
    @tool
    async def show_product_photos(product_ids: List[int]) -> str:
        """Отправляет фотографии товаров клиенту через WhatsApp.

        Отправляет фотографии товаров по их ID напрямую клиенту через WhatsApp API.
        Товары ищутся в базе данных по ID, фотографии отправляются автоматически.

        ════════════════════════════════════════════════════════════════════════════════
        КОГДА ИСПОЛЬЗОВАТЬ:
        ════════════════════════════════════════════════════════════════════════════════

        ✅ Используй ТОЛЬКО в двух случаях:

        1. Когда пользователь ЯВНО просит показать/отправить фото:
           - "отправь фото", "покажи фото", "фотографии", "покажи фотографии товаров"
           - "отправь фото этих товаров", "хочу увидеть фото"
           - "покажи фото грудинки свиной", "отправь фото товаров от Коралл"

        2. При инициализации разговора (init_conversation) - первое сообщение в диалоге

        ❌ НЕ используй:
        - После обычного поиска товаров БЕЗ явного запроса фото
        - Автоматически после vector_search или execute_sql_request
        - Если пользователь просто спрашивает про товары без упоминания фото

        ════════════════════════════════════════════════════════════════════════════════
        ВАЖНО - ДВА СЦЕНАРИЯ:
        ════════════════════════════════════════════════════════════════════════════════

        СЦЕНАРИЙ 1: Клиент просит фото КОНКРЕТНЫХ товаров
        Пример: "покажи фото грудинки свиной", "отправь фото товаров от Коралл"

        Алгоритм:
        1. СНАЧАЛА найди товары: vector_search("грудинка свиная") или execute_sql_request
        2. Получи ID из ответа: [PRODUCT_IDS]{{"product_ids": [789, 790]}}[/PRODUCT_IDS]
        3. ПОТОМ отправь фото: show_product_photos product_ids=[789, 790]
        4. НЕ используй ID из старых запросов!
        5. ВАЖНО: При обычных запросах отправляется только 1 фото (первый товар из списка)

        СЦЕНАРИЙ 2: Клиент просто просит "отправь фото" (без уточнения)
        Пример: После поиска "есть коралл" → клиент: "отправь фото"

        Алгоритм:
        1. Используй ID из ПОСЛЕДНЕГО ответа инструментов поиска в chat_history
        2. Извлеки product_ids из секции [PRODUCT_IDS] из последнего ответа
        3. show_product_photos product_ids=[извлеченные ID]

        Args:
            product_ids: Список ID товаров для отправки фото (извлеки из секции [PRODUCT_IDS] ответа инструментов поиска)

        Returns:
            Строка с статусом отправки: количество отправленных, не отправленных и не найденных товаров
        """
        logger.info(f"[show_product_photos] Sending photos to {client_phone} for product_ids: {product_ids}")

        if not product_ids:
            return "Нет товаров для отправки фотографий."

        if is_init_message:
            product_ids = product_ids[:2]
            logger.info(f"[show_product_photos] Init conversation: ограничено до 2 товаров")
        else:
            product_ids = product_ids[:1]
            logger.info(f"[show_product_photos] Обычный запрос: ограничено до 1 товара")

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
                f"Товары найдены в базе данных, но фотографии либо отсутствуют, либо произошла ошибка при отправке."
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
