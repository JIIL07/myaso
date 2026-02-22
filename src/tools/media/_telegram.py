"""Shared Telegram file-sending helper for media tools."""

from __future__ import annotations

from src.config.settings import settings
from src.constants import HTTP_TIMEOUT_SECONDS
from src.toolkit import has_client_phone
from src.utils.http.http_client import send_http_post


async def send_telegram_file(
    phone: str,
    file_url: str,
    caption: str,
    extension: str = "png",
) -> bool:
    """Send a file to a Telegram user via the bot API proxy."""
    if not has_client_phone(phone):
        return False

    if not file_url or not str(file_url).strip():
        return False

    return await send_http_post(
        url=settings.telegram.send_file_url,
        payload={
            "recipient": phone,
            "file_url": file_url,
            "caption": caption,
            "extension": extension,
        },
        timeout=HTTP_TIMEOUT_SECONDS,
        service_name="telegram",
        recipient=phone,
        additional_context="файл: %s" % file_url,
    )

