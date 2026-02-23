from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


async def send_http_post_multipart(
    url: str,
    fields: dict[str, str],
    timeout: float,
    service_name: str = "http",
    recipient: str | None = None,
    additional_context: str | None = None,
) -> bool:
    """POST multipart/form-data (for /send-file etc.)."""
    try:
        files = [(k, (None, v)) for k, v in fields.items() if v is not None and str(v).strip()]
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url=url, files=files)
            response.raise_for_status()
            return True
    except httpx.HTTPStatusError as e:
        context_parts = ["HTTP ошибка для %s" % recipient] if recipient else ["HTTP ошибка"]
        if additional_context:
            context_parts.append(", %s" % additional_context)
        context_parts.append(": статус %s" % e.response.status_code)
        logger.error("[%s] %s", service_name, "".join(context_parts))
        return False
    except httpx.TimeoutException:
        context = "для %s" % recipient if recipient else ""
        if additional_context:
            context += ", %s" % additional_context
        logger.error("[%s] Таймаут при отправке %s", service_name, context)
        return False
    except Exception as e:
        context = "для %s" % recipient if recipient else ""
        if additional_context:
            context += ", %s" % additional_context
        logger.error("[%s] Ошибка %s: %s", service_name, context, e, exc_info=True)
        return False


async def send_http_post(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    service_name: str = "http",
    recipient: str | None = None,
    additional_context: str | None = None,
) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url=url, json=payload)
            response.raise_for_status()
            return True
    except httpx.HTTPStatusError as e:
        context_parts = ["HTTP ошибка для %s" % recipient] if recipient else ["HTTP ошибка"]
        if additional_context:
            context_parts.append(", %s" % additional_context)
        context_parts.append(": статус %s" % e.response.status_code)

        logger.error(
            "[%s] %s", service_name, "".join(context_parts)
        )
        return False
    except httpx.TimeoutException:
        context = "для %s" % recipient if recipient else ""
        if additional_context:
            context += ", %s" % additional_context

        logger.error(
            "[%s] Таймаут при отправке %s", service_name, context
        )
        return False
    except Exception as e:
        context = "для %s" % recipient if recipient else ""
        if additional_context:
            context += ", %s" % additional_context

        logger.error(
            "[%s] Ошибка %s: %s", service_name, context, e,
            exc_info=True,
        )
        return False
