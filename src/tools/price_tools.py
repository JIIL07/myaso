"""Инструменты для расчета цен товаров."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Tuple

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.services.ai.prompt import get_all_system_values
from src.utils.prices.price_calculator import calculate_final_price
from src.services.database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


@tool(response_format="content_and_artifact")
async def calculate_product_price(
    order_price_kg: float,
    supplier_name: Optional[str] = None,
    runtime: ToolRuntime[ProductAgentContext, ProductAgentState] = None,
) -> Tuple[str, Dict]:
    """Рассчитывает финальную цену товара с учетом наценок.

    Для товаров поставщика "ООО КИТ" цена уже финальная в БД, наценки не применяются.

    Args:
        order_price_kg: Базовая цена за кг из БД (может быть 0, None, float, или строка)
        supplier_name: Название поставщика (опционально).
                       Если поставщик "ООО КИТ" - цена возвращается без изменений
        runtime: ToolRuntime для доступа к context и state

    Returns:
        Кортеж из двух элементов:
        - Финальная цена как строка: "Цена по запросу" если order_price_kg == 0/None/пустая строка,
          иначе цена с учетом наценок (округленная до 2 знаков) или без изменений для "ООО КИТ"
        - Словарь с данными о цене: price, order_price_kg, supplier_name
    """
    try:
        system_vars = await get_all_system_values()
        final_price = calculate_final_price(
            order_price_kg, system_vars, supplier_name=supplier_name
        )
        artifact = {
            "price": final_price,
            "order_price_kg": order_price_kg,
            "supplier_name": supplier_name,
        }
        return final_price, artifact
    except Exception as e:
        logger.error(
            "[calculate_product_price] Ошибка расчета цены: %s",
            e,
            exc_info=True,
        )
        error_text = "Цена по запросу"
        return error_text, {
            "price": error_text,
            "order_price_kg": order_price_kg,
            "supplier_name": supplier_name,
        }
