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


@tool
async def show_product_photos(product_titles: List[str], phone: str) -> str:
    """Отправляет фотографии товаров пользователю через WhatsApp API.

    Находит товары по их названиям в базе данных и отправляет фотографии
    на указанный номер телефона через WhatsApp API.

    Используй этот инструмент когда:
    - Пользователь просит показать фото товаров
    - Нужно визуально представить товары пользователю
    - После поиска товаров, если пользователь хочет увидеть фото

    Args:
        product_titles: Список названий товаров, фото которых нужно отправить
        phone: Номер телефона пользователя в формате WhatsApp

    Returns:
        Строка с статусом отправки фотографий (какие отправлены, каких нет)
    """
    if not product_titles:
        return "Нет товаров для отправки фотографий."

    has_photo = []
    no_photo = []
    not_found = []

    supabase = await get_supabase_client()

    for title in product_titles:
        try:
            result = (
                await supabase.table("products")
                .select("*")
                .eq("title", title)
                .execute()
            )

            if not result.data or len(result.data) == 0:
                not_found.append(title)
                continue

            found_with_photo = False
            for product in result.data:
                photo_url = product.get("photo")

                if photo_url:
                    try:
                        async with httpx.AsyncClient(
                            timeout=HTTP_TIMEOUT_SECONDS
                        ) as client:
                            response = await client.post(
                                url=settings.whatsapp.send_image_url,
                                json={
                                    "recipient": phone,
                                    "image_url": photo_url,
                                    "caption": title,
                                },
                            )
                            response.raise_for_status()
                        found_with_photo = True
                    except Exception as e:
                        logger.warning(
                            f"[show_product_photos] Ошибка отправки фото для {title}: {e}"
                        )

            if found_with_photo:
                has_photo.append(title)
            else:
                no_photo.append(title)
        except Exception as e:
            logger.error(
                f"[show_product_photos] Ошибка при поиске товара '{title}': {e}"
            )
            not_found.append(title)

    result_parts = []
    if has_photo:
        result_parts.append(
            f"Фотографии следующих товаров отправлены: {', '.join(has_photo)}"
        )
    if no_photo:
        result_parts.append(f"Нет фотографий следующих товаров: {', '.join(no_photo)}")
    if not_found:
        result_parts.append(f"Товары не найдены: {', '.join(not_found)}")

    return (
        "\n".join(result_parts)
        if result_parts
        else "Нет товаров для отправки фотографий."
    )

