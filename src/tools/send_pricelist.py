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
from src.tools._telegram import send_telegram_file

logger = logging.getLogger(__name__)


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
            return "Номер телефона клиента не указан.", {"success": False, "error": "missing_phone"}

        # --- Fetch pricelist URL from system table ---
        try:
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
            pricelist_url = dict(row).get(COLUMN_VALUE) if row else None
        except Exception as e:
            logger.error("[send_pricelist] Ошибка получения URL: %s", e, exc_info=True)
            return (
                "Произошла ошибка при получении прайс-листа. Попробуйте позже."
            ), {"success": False, "error": "system_error"}

        if not pricelist_url or not str(pricelist_url).strip():
            return (
                "Прайс-лист не настроен в системе. "
                "Сообщи клиенту, что прайс-лист временно недоступен."
            ), {"success": False, "error": "not_configured"}

        # --- Determine file extension ---
        try:
            parsed_path = urlparse(str(pricelist_url)).path
            file_extension = os.path.splitext(parsed_path)[1].lstrip(".").lower() or "xlsx"
        except Exception:
            file_extension = "xlsx"

        # --- Send ---
        ok = await send_telegram_file(
            phone=client_phone,
            file_url=str(pricelist_url),
            caption=SYSTEM_VALUE_PRICELIST,
            extension=file_extension,
        )

        if ok:
            return (
                "✅ Прайс-лист успешно отправлен клиенту через Telegram."
            ), {"success": True, "url": str(pricelist_url), "extension": file_extension}

        return (
            "❌ Не удалось отправить прайс-лист. "
            "Сообщи клиенту, что он временно недоступен, но ты можешь помочь с поиском товаров."
        ), {"success": False, "error": "send_failed"}

    except Exception as e:
        logger.error("[send_pricelist] Критическая ошибка: %s", e, exc_info=True)
        return (
            "❌ Ошибка при отправке прайс-листа. "
            "Сообщи клиенту, что он временно недоступен, но ты можешь помочь с поиском товаров."
        ), {"success": False, "error": "critical_error"}
