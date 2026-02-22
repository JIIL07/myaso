"""Tool: show_product_photos — send product photos via Telegram."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.services.database.database import get_pool
from src.toolkit import has_client_phone
from src.tools._contract import fail_response, ok_response
from src.tools._telegram import send_telegram_file

logger = logging.getLogger(__name__)

_EMPTY_ARTIFACT: dict[str, Any] = {"sent": [], "failed": [], "not_found": [], "total": 0}


async def _fetch_products_map(product_ids: list[int]) -> dict[int, dict[str, Any]]:
    normalized_ids = [int(product_id) for product_id in product_ids]
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, photo
            FROM myaso.products
            WHERE id = ANY($1::int[])
            """,
            normalized_ids,
        )
    return {row["id"]: dict(row) for row in rows}


def _build_result_text(sent_ids: list[int], failed_ids: list[int], not_found_ids: list[int]) -> str:
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
    return "\n\n".join(parts) if parts else "Нет товаров для отправки фотографий."


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

        if not has_client_phone(client_phone):
            return fail_response(
                "Номер телефона клиента не указан.",
                error_code="missing_phone",
                artifact=_EMPTY_ARTIFACT.copy(),
            )

        product_ids = runtime.state.get("product_ids", [])

        if not product_ids:
            return fail_response(
                "Нет сохраненных ID товаров для отправки фотографий. "
                "Сначала используй инструменты поиска товаров.",
                error_code="missing_product_ids",
                artifact=_EMPTY_ARTIFACT.copy(),
            )

        sent_ids: list[int] = []
        failed_ids: list[int] = []
        not_found_ids: list[int] = []

        try:
            products_map = await _fetch_products_map(product_ids)
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

        result_text = _build_result_text(sent_ids, failed_ids, not_found_ids)
        artifact = {
            "sent": sent_ids,
            "failed": failed_ids,
            "not_found": not_found_ids,
            "total": len(product_ids),
            "sent_count": len(sent_ids),
            "failed_count": len(failed_ids),
            "not_found_count": len(not_found_ids),
        }
        if sent_ids:
            return ok_response(result_text, artifact=artifact)
        return fail_response(result_text, error_code="nothing_sent", artifact=artifact)

    except Exception as e:
        logger.error("[show_product_photos] Критическая ошибка: %s", e, exc_info=True)
        return fail_response(
            "Произошла критическая ошибка при отправке фотографий. "
            "Попробуйте позже или обратитесь в поддержку.",
            error_code="critical_error",
            artifact=_EMPTY_ARTIFACT.copy(),
        )
