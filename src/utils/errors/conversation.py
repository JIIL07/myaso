import logging
from typing import Any, Dict

from src.services.ai.prompt import get_prompt
from src.services.langfuse.prompt_names import (
    PROMPT_NAME_ERROR_HANDLER,
    PROMPT_NAME_HUMAN_IN_THE_LOOP,
)


logger = logging.getLogger(__name__)


async def handle_conversation_error(
    client_phone: str,
    context: str,
    error: Exception,
    messaging_service: Any,
) -> Dict[str, Any]:
    logger.error(
        f"[{context}] Ошибка обработки для {client_phone}: {error}",
        exc_info=True,
    )

    hitl_prompt = await get_prompt(
        prompt_name=PROMPT_NAME_HUMAN_IN_THE_LOOP,
        default_prompt=(
            "Извините, произошла ошибка при обработке вашего запроса. "
            "Пожалуйста, свяжитесь с нашим менеджером для получения помощи."
        ),
    )

    if hitl_prompt:
        try:
            await messaging_service.send_text_message(
                client_phone=client_phone,
                message_text=hitl_prompt,
                context=context,
            )
            return {"success": False, "error": str(error)}
        except Exception as send_error:
            logger.warning(
                f"[{context}] Не удалось отправить HITL промпт для {client_phone}: {send_error}"
            )

    error_prompt = await get_prompt(
        prompt_name=PROMPT_NAME_ERROR_HANDLER,
        default_prompt="Произошла ошибка при обработке запроса.",
    )

    if error_prompt:
        try:
            await messaging_service.send_text_message(
                client_phone=client_phone,
                message_text=error_prompt,
                context=context,
            )
            return {"success": False, "error": str(error)}
        except Exception as send_error:
            logger.warning(
                f"[{context}] Не удалось отправить error handler промпт для {client_phone}: {send_error}"
            )

    try:
        await messaging_service.send_error_message(client_phone, context)
    except Exception as send_error:
        logger.error(
            f"[{context}] Не удалось отправить стандартное сообщение об ошибке для {client_phone}: {send_error}",
            exc_info=True,
        )

    return {"success": False, "error": str(error)}

