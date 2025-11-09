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
        КОГДА ИСПОЛЬЗОВАТЬ (ТРИГГЕРЫ):
        ════════════════════════════════════════════════════════════════════════════════

        ВЫЗЫВАЙ этот инструмент когда клиент использует следующие слова/фразы:
        - "отправь фото", "отправь фотографии", "отправь фото товаров"
        - "покажи фото", "покажи фотографии", "покажи фото товаров"
        - "фото", "фотографии" (в контексте просьбы показать)
        - "хочу увидеть фото", "хочу посмотреть фото"
        - "отправь фото грудинки", "покажи фото коралла", "фото товаров от Коралл"
        - ЛЮБОЙ запрос содержащий слова "отправь" + "фото" или "покажи" + "фото"

        1. ОБЯЗАТЕЛЬНО при инициализации разговора (init_conversation):
           - Это первое сообщение в диалоге
           - После того как ты нашел товары через execute_sql_request или vector_search
           - ОБЯЗАТЕЛЬНО вызови этот инструмент после поиска товаров!
           - Извлеки product_ids из секции [PRODUCT_IDS] ответа инструмента поиска
           - ВАЖНО: Возьми ТОЛЬКО первые 2 ID (даже если найдено больше товаров)
           - Вызови show_product_photos(product_ids=[id1, id2]) с максимум 2 ID

        2. Когда пользователь просит показать/отправить фото (слова-триггеры выше):
           - СНАЧАЛА найди товары через vector_search или generate_sql_from_text + execute_sql_request
           - Покажи найденные товары клиенту
           - Извлеки product_ids из секции [PRODUCT_IDS] в ответе инструмента поиска
           - ВАЖНО: Если найдено много товаров (например, 48 товаров):
             * Возьми ТОЛЬКО первые 1-3 ID (максимум 3!)
             * НЕ передавай все найденные ID!
             * Вызови show_product_photos(product_ids=[id1, id2, id3]) с максимум 3 ID
           - Если клиент просто сказал "отправь фото" без уточнения:
             * Используй product_ids из последнего ответа инструментов поиска в chat_history

        ════════════════════════════════════════════════════════════════════════════════
        ВАЖНО - ОГРАНИЧЕНИЯ КОЛИЧЕСТВА ФОТО:
        ════════════════════════════════════════════════════════════════════════════════
        
        - При обычных запросах отправляется МАКСИМУМ 1-3 фото (первые товары из списка)
        - При init_conversation отправляется МАКСИМУМ 2 фото (первые два товара)
        - Если найдено много товаров (например, 48 товаров) - передай ТОЛЬКО первые 1-3 ID!
        - НЕ передавай все найденные ID, даже если их много!
        - Отправляются только товары, у которых есть фотография в базе данных
        - ID товаров должны быть извлечены из секции [PRODUCT_IDS] ответа инструментов поиска

        ПРИМЕРЫ ПРАВИЛЬНОГО ИСПОЛЬЗОВАНИЯ:
        - Найдено 48 товаров с ID [1, 2, 3, ..., 48] → передай ТОЛЬКО [1] или [1, 2, 3] (максимум 3)
        - Найдено 5 товаров с ID [10, 20, 30, 40, 50] → передай ТОЛЬКО [10] или [10, 20] (максимум 3)
        - Найдено 2 товара с ID [100, 200] → можно передать [100, 200]

        ПРИМЕРЫ НЕПРАВИЛЬНОГО ИСПОЛЬЗОВАНИЯ (НИКОГДА ТАК НЕ ДЕЛАЙ!):
        - Найдено 48 товаров → НЕ передавай все 48 ID! Передай только первые 1-3
        - show_product_photos(product_ids=[1, 2, 3, ..., 48]) → НЕПРАВИЛЬНО!

        Args:
            product_ids: Список ID товаров для отправки фото (МАКСИМУМ 3 ID!). 
            ОБЯЗАТЕЛЬНО извлеки из секции [PRODUCT_IDS] ответа инструментов поиска (vector_search, execute_sql_request). 
            Если найдено много товаров - передай ТОЛЬКО первые 1-3 ID! НЕ передавай все найденные ID!
            НЕ используй ID из старых запросов.

        Returns:
            Строка с результатом отправки фотографий через WhatsApp
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
