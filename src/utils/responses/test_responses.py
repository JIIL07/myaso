import logging
from typing import Any, Awaitable, Callable, Dict

from src.entities import TestResponse
from src.utils.logger.masking import mask_phone


logger = logging.getLogger(__name__)


async def format_test_response(
    service_method: Callable[[Any], Awaitable[Dict[str, Any]]],
    request: Any,
    context: str,
) -> TestResponse:
    try:
        result = await service_method(request)

        if result.get("success"):
            logger.info(
                f"[{context}] Успешно обработано для {mask_phone(request.client_phone)}"
            )
            return TestResponse(
                success=True,
                response_text=result.get("response_text"),
            )

        logger.warning(
            f"[{context}] Ошибка обработки для {mask_phone(request.client_phone)}: {result.get('error')}"
        )
        return TestResponse(
            success=False,
            error=result.get("error", "Unknown error"),
        )
    except Exception as e:
        logger.error(
            f"[{context}] Критическая ошибка для {mask_phone(request.client_phone)}: {e}",
            exc_info=True,
        )
        return TestResponse(
            success=False,
            error=str(e),
        )


def format_queue_response(
    result: Dict[str, Any],
    client_phone: str,
    context: str,
) -> Dict[str, Any]:
    if result.get("queued"):
        logger.info(
            f"[{context}] Запрос добавлен в очередь для {mask_phone(client_phone)}"
        )
    else:
        logger.info(f"[{context}] Запрос обработан для {mask_phone(client_phone)}")

    return result

