"""Tool: show_product_photos — send product photos via Telegram."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout
from src.tools._telegram import send_telegram_file
from src.utils.validators import validate_client_phone

logger = logging.getLogger(__name__)

_EMPTY_ARTIFACT: dict[str, Any] = {"sent": [], "failed": [], "not_found": [], "total": 0}


@tool(response_format="content_and_artifact")
async def show_product_photos(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> tuple[str, dict[str, Any]]:
    """Отправляет фотографии найденных товаров клиенту через Telegram.

    Использует product_ids из state агента (заполняются инструментами поиска).

    КОГДА ИСПОЛЬЗОВАТЬ:
    - "покажи фото", "есть ли фото", "фотографии товаров"
    - После поиска товаров — визуальная демонстрация

    НЕ ИСПОЛЬЗОВАТЬ:
    - Не было поиска товаров (нет product_ids в state)
    - Клиент не просит фотографии
    """
    try:
        client_phone = runtime.context.client_phone

        if not validate_client_phone(client_phone):
            return "Номер телефона клиента не указан.", _EMPTY_ARTIFACT.copy()

        product_ids = runtime.state.get("product_ids", [])

        if not product_ids:
            return (
                "Нет сохраненных ID товаров для отправки фотографий. "
                "Сначала используй инструменты поиска товаров."
            ), _EMPTY_ARTIFACT.copy()

        sent_ids: list[int] = []
        failed_ids: list[int] = []
        not_found_ids: list[int] = []

        supabase = await get_supabase_client()

        try:
            result = await execute_with_timeout(
                supabase.table("products")
                .select("id, title, photo")
                .in_("id", product_ids)
                .execute(),
                operation_name="show_product_photos.batch_get_products",
            )
            products_map = {p["id"]: p for p in (result.data or [])}
        except Exception as e:
            logger.error("[show_product_photos] Ошибка получения товаров: %s", e, exc_info=True)
            not_found_ids = product_ids.copy()
            products_map = {}

        for product_id in product_ids:
            try:
                product = products_map.get(product_id)
                if not product:
                    not_found_ids.append(product_id)
                    continue

                photo_url = product.get("photo")
                product_title = product.get("title", "Товар #%s" % product_id)

                if not photo_url or not str(photo_url).strip():
                    failed_ids.append(product_id)
                    continue

                ok = await send_telegram_file(
                    phone=client_phone,
                    file_url=str(photo_url),
                    caption=product_title,
                    extension="png",
                )
                (sent_ids if ok else failed_ids).append(product_id)

            except Exception as e:
                logger.error(
                    "[show_product_photos] Ошибка товара ID %s: %s",
                    product_id,
                    e,
                    exc_info=True,
                )
                failed_ids.append(product_id)

        # --- Build result text ---
        parts: list[str] = []
        if sent_ids:
            parts.append(
                "✅ УСПЕШНО ОТПРАВЛЕНО: Фотографии %d товаров отправлены клиенту." % len(sent_ids)
            )
        if failed_ids:
            parts.append(
                "❌ НЕ ОТПРАВЛЕНО: %d товаров — фото отсутствует или ошибка отправки. "
                "Предоставь информацию текстом." % len(failed_ids)
            )
        if not_found_ids:
            parts.append(
                "⚠️ НЕ НАЙДЕНО: %d товаров не найдены в базе данных." % len(not_found_ids)
            )

        result_text = "\n\n".join(parts) if parts else "Нет товаров для отправки фотографий."

        artifact = {
            "sent": sent_ids,
            "failed": failed_ids,
            "not_found": not_found_ids,
            "total": len(product_ids),
            "sent_count": len(sent_ids),
            "failed_count": len(failed_ids),
            "not_found_count": len(not_found_ids),
        }
        return result_text, artifact

    except Exception as e:
        logger.error("[show_product_photos] Критическая ошибка: %s", e, exc_info=True)
        return (
            "Произошла критическая ошибка при отправке фотографий. "
            "Попробуйте позже или обратитесь в поддержку."
        ), _EMPTY_ARTIFACT.copy()
