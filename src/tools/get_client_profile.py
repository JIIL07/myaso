"""Tool: get_client_profile — retrieve client profile information."""

from __future__ import annotations

import logging
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.queries.clients_queries import get_client_profile_text
from src.toolkit import has_client_phone
from src.tools._contract import fail_response, ok_response

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def get_client_profile(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> tuple[str, dict[str, Any]]:
    """Профиль клиента: контакты, стиль общения, предпочтения."""
    try:
        client_phone = runtime.context.client_phone

        if not has_client_phone(client_phone):
            return fail_response("Номер телефона клиента не указан.", error_code="missing_phone")

        profile_text = await get_client_profile_text(client_phone)
        artifact = {
            "phone": client_phone,
            "profile_retrieved": bool(profile_text and profile_text.strip()),
        }
        return ok_response(profile_text, artifact=artifact)
    except Exception as e:
        logger.error("[get_client_profile] Ошибка: %s", e, exc_info=True)
        return fail_response(
            "Произошла ошибка при получении профиля клиента. "
            "Попробуйте позже или обратитесь в поддержку.",
            error_code="profile_error",
        )
