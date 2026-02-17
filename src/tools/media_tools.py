"""Инструменты для отправки файлов и медиа-сообщений клиенту."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.services.database.constants import COLUMN_TOPIC, COLUMN_VALUE, TABLE_SYSTEM
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.config.settings import settings
from src.services.ai.constants import SYSTEM_VALUE_PRICELIST
from src.services.telegram.constants import HTTP_TIMEOUT_SECONDS
from src.utils.http.http_client import send_http_post
from src.utils.validators import validate_not_empty, validate_client_phone

logger = logging.getLogger(__name__)


async def _send_telegram_file(
    phone: str,
    file_url: str,
    caption: str,
    extension: str = "png",
) -> bool:
    """Отправляет файл в Telegram по URL, возвращает признак успеха."""
    if not validate_not_empty(phone, "ID чата", "_send_telegram_file"):
        return False

    if not validate_not_empty(file_url, "URL файла", "_send_telegram_file"):
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
        additional_context=f"файл: {file_url}",
    )


@tool(response_format="content_and_artifact")
async def show_product_photos(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> Tuple[str, Dict]:
    """Отправляет клиенту фото товаров из текущего состояния агента и возвращает статистику отправки."""
    try:
        client_phone = runtime.context.client_phone

        if not validate_client_phone(client_phone):
            error_text = "Номер телефона клиента не указан."
            return error_text, {
                "sent": [],
                "failed": [],
                "not_found": [],
                "total": 0,
            }

        product_ids = runtime.state.get("product_ids", [])

        if not product_ids:
            error_text = (
                "Нет сохраненных ID товаров для отправки фотографий. "
                "Используйте инструменты поиска товаров "
                "(vector_search, execute_sql_query, get_random_products, get_product_by_title) "
                "для поиска товаров."
            )
            return error_text, {
                "sent": [],
                "failed": [],
                "not_found": [],
                "total": 0,
            }

        sent_ids: List[int] = []
        failed_ids: List[int] = []
        not_found_ids: List[int] = []

        supabase = await get_supabase_client()

        # Batch query вместо N+1 запросов
        try:
            result = await execute_with_timeout(
                supabase.table("products")
                .select("id, title, photo")
                .in_("id", product_ids)
                .execute(),
                operation_name="show_product_photos.batch_get_products",
            )

            products_map = {p["id"]: p for p in (result.data or [])}
        except Exception as e:
            logger.error(
                "[show_product_photos] Ошибка при получении товаров: %s",
                e,
                exc_info=True,
            )
            # Если batch запрос не удался, помечаем все как not_found
            not_found_ids = product_ids.copy()
            products_map = {}

        # Обработка каждого товара
        for product_id in product_ids:
            try:
                product = products_map.get(product_id)
                if not product:
                    not_found_ids.append(product_id)
                    continue

                photo_url = product.get("photo")
                product_title = product.get("title", f"Товар #{product_id}")

                if not photo_url or not str(photo_url).strip():
                    failed_ids.append(product_id)
                    continue

                send_success = await _send_telegram_file(
                    phone=client_phone,
                    file_url=str(photo_url),
                    caption=product_title,
                    extension="png",
                )

                if send_success:
                    sent_ids.append(product_id)
                else:
                    failed_ids.append(product_id)

            except Exception as e:
                logger.error(
                    "[show_product_photos] Ошибка при обработке товара ID %s: %s",
                    product_id,
                    e,
                    exc_info=True,
                )
                failed_ids.append(product_id)

        result_parts: List[str] = []

        if sent_ids:
            result_parts.append(
                f"✅ УСПЕШНО ОТПРАВЛЕНО: Фотографии {len(sent_ids)} товаров "
                "успешно отправлены клиенту через Telegram. "
                "Клиент получил эти фотографии."
            )

        if failed_ids:
            result_parts.append(
                f"❌ НЕ ОТПРАВЛЕНО: Не удалось отправить фотографии {len(failed_ids)} товаров. "
                "Товары найдены в базе данных, но фотографии либо отсутствуют, "
                "либо произошла ошибка при отправке. Предоставь информацию о товарах текстом."
            )

        if not_found_ids:
            result_parts.append(
                f"⚠️ НЕ НАЙДЕНО: {len(not_found_ids)} товаров не найдены в базе данных. "
                "Эти товары отсутствуют в каталоге."
            )

        result_text = (
            "\n\n".join(result_parts)
            if result_parts
            else "Нет товаров для отправки фотографий."
        )

        artifact = {
            "sent": sent_ids,
            "failed": failed_ids,
            "not_found": not_found_ids,
            "total": len(product_ids),
            "sent_count": len(sent_ids),
            "failed_count": len(failed_ids),
            "not_found_count": len(not_found_ids),
        }

        return result_text, artifact

    except Exception as e:
        logger.error(
            "[show_product_photos] Критическая ошибка: %s",
            e,
            exc_info=True,
        )
        error_text = (
            "Произошла критическая ошибка при отправке фотографий. "
            "Попробуйте позже или обратитесь в поддержку."
        )
        return error_text, {
            "sent": [],
            "failed": [],
            "not_found": [],
            "total": 0,
        }


@tool(response_format="content_and_artifact")
async def send_pricelist(
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState],
) -> Tuple[str, Dict]:
    """Отправляет клиенту прайс-лист по URL из системных настроек и возвращает результат отправки."""
    try:
        client_phone = runtime.context.client_phone

        if not validate_client_phone(client_phone):
            error_text = "Номер телефона клиента не указан."
            return error_text, {"success": False, "error": "missing_phone"}

        try:
            supabase = await get_supabase_client()
            result = await execute_with_timeout(
                supabase.table(TABLE_SYSTEM)
                .select(COLUMN_VALUE)
                .eq(COLUMN_TOPIC, SYSTEM_VALUE_PRICELIST)
                .execute(),
                operation_name="send_pricelist.get_pricelist_url",
            )
            pricelist_url = result.data[0].get(COLUMN_VALUE) if result.data else None
        except Exception as e:
            logger.error(
                "[send_pricelist] Ошибка при получении URL прайс-листа: %s",
                e,
                exc_info=True,
            )
            error_text = (
                "Произошла ошибка при получении прайс-листа из системы. "
                "Попробуйте позже или обратитесь в поддержку."
            )
            return error_text, {"success": False, "error": "system_error"}

        if not pricelist_url or not str(pricelist_url).strip():
            error_text = (
                "Прайс-лист не настроен в системе. "
                "Сообщи клиенту, что прайс-лист временно недоступен."
            )
            return error_text, {"success": False, "error": "not_configured"}

        try:
            parsed_path = urlparse(str(pricelist_url)).path
            file_extension = (
                os.path.splitext(parsed_path)[1].lstrip(".").lower() or "xlsx"
            )
        except Exception:
            file_extension = "xlsx"

        send_success = await _send_telegram_file(
            phone=client_phone,
            file_url=str(pricelist_url),
            caption=SYSTEM_VALUE_PRICELIST,
            extension=file_extension,
        )

        if send_success:
            success_text = (
                "✅ Прайс-лист успешно отправлен клиенту через Telegram. "
                "Клиент получил файл с прайс-листом."
            )
            return success_text, {
                "success": True,
                "url": str(pricelist_url),
                "extension": file_extension,
            }
        else:
            error_text = (
                "❌ Не удалось отправить прайс-лист. "
                "Сообщи клиенту, что прайс-лист временно недоступен, "
                "но ты можешь помочь с поиском товаров."
            )
            return error_text, {"success": False, "error": "send_failed"}

    except Exception as e:
        logger.error(
            "[send_pricelist] Критическая ошибка: %s",
            e,
            exc_info=True,
        )
        error_text = (
            "❌ Ошибка при отправке прайс-листа. "
            "Сообщи клиенту, что прайс-лист временно недоступен, "
            "но ты можешь помочь с поиском товаров."
        )
        return error_text, {"success": False, "error": "critical_error"}
