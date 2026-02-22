"""Tool: send_pricelist — send the company pricelist via Telegram."""

from __future__ import annotations

import logging
import os
from typing import Any
from urllib.parse import urlparse

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import COLUMN_TOPIC, COLUMN_VALUE, TABLE_SYSTEM, SYSTEM_VALUE_PRICELIST
from src.services.database.database import get_pool
from src.toolkit import has_client_phone
from src.tools._contract import fail_response, ok_response
from src.tools._telegram import send_telegram_file

logger = logging.getLogger(__name__)


def _extract_file_extension(file_url: str) -> str:
    try:
        parsed_path = urlparse(file_url).path
        return os.path.splitext(parsed_path)[1].lstrip(".").lower() or "xlsx"
    except Exception:
        return "xlsx"


async def _get_pricelist_url() -> str | None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT %s
            FROM myaso.%s
            WHERE %s = $1
            LIMIT 1
            """
            % (COLUMN_VALUE, TABLE_SYSTEM, COLUMN_TOPIC),
            SYSTEM_VALUE_PRICELIST,
        )
    return dict(row).get(COLUMN_VALUE) if row else None


@tool(response_format="content_and_artifact")
async def send_pricelist(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> tuple[str, dict[str, Any]]:
    """Отправляет прайс-лист клиенту через Telegram (обычно Excel-файл).

    КОГДА ИСПОЛЬЗОВАТЬ:
    - "прайс-лист", "прайс", "файл с ценами", "пришли прайс"

    НЕ ИСПОЛЬЗОВАТЬ:
    - Вопрос о конкретных товарах -> инструменты поиска
    - Клиент не просит файл явно
    """
    try:
        client_phone = runtime.context.client_phone

        if not has_client_phone(client_phone):
            return fail_response("Номер телефона клиента не указан.", error_code="missing_phone")

        # --- Fetch pricelist URL from system table ---
        try:
            pricelist_url = await _get_pricelist_url()
        except Exception as e:
            logger.error("[send_pricelist] Ошибка получения URL: %s", e, exc_info=True)
            return fail_response(
                "Произошла ошибка при получении прайс-листа. Попробуйте позже.",
                error_code="system_error",
            )

        if not pricelist_url or not str(pricelist_url).strip():
            return fail_response(
                "Прайс-лист не настроен в системе. "
                "Сообщи клиенту, что прайс-лист временно недоступен.",
                error_code="not_configured",
            )

        file_extension = _extract_file_extension(str(pricelist_url))

        # --- Send ---
        ok = await send_telegram_file(
            phone=client_phone,
            file_url=str(pricelist_url),
            caption=SYSTEM_VALUE_PRICELIST,
            extension=file_extension,
        )

        if ok:
            return ok_response(
                "✅ Прайс-лист успешно отправлен клиенту через Telegram.",
                artifact={"url": str(pricelist_url), "extension": file_extension},
            )

        return fail_response(
            "❌ Не удалось отправить прайс-лист. "
            "Сообщи клиенту, что он временно недоступен, но ты можешь помочь с поиском товаров.",
            error_code="send_failed",
        )

    except Exception as e:
        logger.error("[send_pricelist] Критическая ошибка: %s", e, exc_info=True)
        return fail_response(
            "❌ Ошибка при отправке прайс-листа. "
            "Сообщи клиенту, что он временно недоступен, но ты можешь помочь с поиском товаров.",
            error_code="critical_error",
        )
