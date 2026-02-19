import logging
from typing import Any

from src.utils.logger.masking import mask_phone

logger = logging.getLogger(__name__)


def format_queue_response(
    result: dict[str, Any],
    client_phone: str,
    context: str,
) -> dict[str, Any]:
    if result.get("queued"):
        logger.info("[%s] Request queued for %s", context, mask_phone(client_phone))
    else:
        logger.info("[%s] Request processed for %s", context, mask_phone(client_phone))
    return result
