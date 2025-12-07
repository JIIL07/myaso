"""Инструменты для расчета цен товаров."""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

from src.utils.price_calculator import calculate_final_price
from src.utils.prompts import get_all_system_values

logger = logging.getLogger(__name__)


@tool
async def calculate_product_price(
    order_price_kg: float,
    supplier_name: Optional[str] = None
) -> str:
    """Рассчитывает финальную цену товара с учетом наценок.
    
    ВАЖНО: Для товаров поставщика "ООО КИТ" цена уже финальная в БД, наценки не применяются.
    
    Используй этот инструмент когда нужно:
    - Рассчитать финальную цену для товара
    - Показать цену с учетом наценок клиенту
    - Сравнить цены разных товаров
    
    Args:
        order_price_kg: Базовая цена за кг из БД (может быть 0, None, float, или строка)
        supplier_name: Название поставщика (опционально). 
                       Если поставщик "ООО КИТ" - цена возвращается без изменений 
                       (цена уже финальная, наценки не применяются)
    
    Returns:
        Финальная цена как строка:
        - "Цена по запросу" если order_price_kg == 0, None, или пустая строка
        - Цена из БД без изменений (округленная до 2 знаков) если поставщик "ООО КИТ" 
          (цена уже финальная, наценки не применяются)
        - Иначе: цена с учетом наценок, округленная до 2 знаков (например, "385.00")
    """
    try:
        system_vars = await get_all_system_values()
        final_price = calculate_final_price(
            order_price_kg,
            system_vars,
            supplier_name=supplier_name
        )
        return final_price
    except Exception as e:
        logger.error(f"[calculate_product_price] Ошибка расчета цены: {e}", exc_info=True)
        return "Цена по запросу"

