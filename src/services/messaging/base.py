import logging
import os
from urllib.parse import urlparse

from src.constants import SYSTEM_VALUE_PRICELIST
from src.services.ai.prompt import get_system_value
from src.toolkit import clean_message_text
from src.utils.http.http_client import send_http_post, send_http_post_multipart

logger = logging.getLogger(__name__)


class BaseMessagingService:
    def __init__(
        self,
        send_message_url: str,
        send_file_url: str,
        timeout: float,
        error_message: str,
        service_name: str,
        recipient_name: str = "получатель",
        message_key: str = "message",
        file_send_as_multipart: bool = False,
    ):
        self.send_message_url = send_message_url
        self.send_file_url = send_file_url
        self.timeout = timeout
        self.error_message = error_message
        self.service_name = service_name
        self.recipient_name = recipient_name
        self.message_key = message_key
        self.file_send_as_multipart = file_send_as_multipart

    async def send_message(self, recipient: str, message: str) -> bool:
        if not recipient or not recipient.strip():
            logger.warning("[Validator] Empty %s (send_message)", self.recipient_name)
            return False
        if not message or not message.strip():
            logger.warning("[Validator] Empty сообщение (send_message)")
            return False
        return await send_http_post(
            url=self.send_message_url,
            payload={"recipient": recipient, self.message_key: message},
            timeout=self.timeout,
            service_name=self.service_name,
            recipient=recipient,
        )

    async def send_image(
        self, recipient: str, file_url: str, caption: str = "", extension: str = "png",
    ) -> bool:
        if not recipient or not recipient.strip():
            logger.warning("[Validator] Empty %s (send_image)", self.recipient_name)
            return False
        if not file_url or not file_url.strip():
            logger.warning("[Validator] Empty URL файла (send_image)")
            return False
        if self.file_send_as_multipart:
            return await send_http_post_multipart(
                url=self.send_file_url,
                fields={"recipient": recipient, "file_url": file_url, "caption": caption},
                timeout=self.timeout,
                service_name=self.service_name,
                recipient=recipient,
                additional_context="файл: %s" % file_url,
            )
        return await send_http_post(
            url=self.send_file_url,
            payload={"recipient": recipient, "file_url": file_url, "caption": caption, "extension": extension},
            timeout=self.timeout,
            service_name=self.service_name,
            recipient=recipient,
            additional_context="файл: %s" % file_url,
        )

    async def send_text_message(
        self, client_id: str, message_text: str, remove_markdown: bool = True, context: str = "",
    ) -> bool:
        try:
            text = clean_message_text(message_text) if remove_markdown else message_text
            success = await self.send_message(client_id, text)
            if not success:
                logger.warning("[%s] Failed to send message for %s", context, client_id)
            return success
        except Exception as e:
            logger.error("[%s] Error sending message for %s: %s", context, client_id, e, exc_info=True)
            return False

    async def send_error_message(self, client_id: str, context: str = "") -> None:
        try:
            await self.send_message(client_id, self.error_message)
        except Exception as e:
            logger.error("[%s] Failed to send error for %s: %s", context, client_id, e, exc_info=True)

    async def send_pricelist(self, client_id: str, context: str = "") -> bool:
        try:
            pricelist_url = await get_system_value(SYSTEM_VALUE_PRICELIST)
            if not pricelist_url:
                return False

            parsed_url = urlparse(pricelist_url)
            _, file_extension = os.path.splitext(parsed_url.path)
            file_extension = file_extension.lstrip(".").lower() if file_extension else "xlsx"

            success = await self.send_image(
                recipient=client_id, file_url=pricelist_url,
                caption="Прайс-лист", extension=file_extension,
            )
            if not success:
                logger.warning("[%s] Failed to send pricelist for %s", context, client_id)
            return success
        except Exception as e:
            logger.error("[%s] Error sending pricelist for %s: %s", context, client_id, e, exc_info=True)
            return False
