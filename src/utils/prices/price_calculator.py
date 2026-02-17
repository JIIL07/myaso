"""Утилиты для расчета финальных цен товаров."""

import logging
import re
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_markup_value(markup_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Парсит значение наценки из строки."""
    if not markup_str:
        return None, None
    
    markup_clean = markup_str.strip()
    has_percent = '%' in markup_clean
    match = re.search(r'(\d+\.?\d*)', markup_clean)
    
    if not match:
        logger.warning(f"Could not parse markup value from: {markup_str}")
        return None, None
    
    value = float(match.group(1))
    
    if has_percent:
        return value, None
    else:
        return None, value


def get_markup_from_system_vars(
    order_price_kg: float,
    system_vars: Dict[str, str]
) -> Tuple[Optional[float], Optional[float]]:
    """Получает наценку из системных переменных на основе цены."""
    if order_price_kg < 100:
        topic = "Наценка на кг/руб (<100 руб)"
    else:
        topic = "Наценка на кг/руб (>100 руб)"
    
    markup_str = system_vars.get(topic)
    
    if not markup_str:
        for key in system_vars.keys():
            key_lower = key.lower()
            if "наценка" in key_lower:
                if "<100" in key_lower and order_price_kg < 100:
                    markup_str = system_vars[key]
                    break
                elif ">100" in key_lower and order_price_kg >= 100:
                    markup_str = system_vars[key]
                    break
        
        if not markup_str:
            for key in system_vars.keys():
                key_lower = key.lower()
                if "наценка" in key_lower and "кг" in key_lower:
                    markup_str = system_vars[key]
                    break
    
    if not markup_str:
        logger.warning(
            f"Markup not found for price {order_price_kg}. "
            f"Available keys: {list(system_vars.keys())}"
        )
        return None, None
    
    return parse_markup_value(markup_str)


def get_delivery_markup(system_vars: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
    """Получает дополнительную наценку из системных переменных."""
    for key in system_vars.keys():
        key_lower = key.lower()
        if ("наценк" in key_lower or "markup" in key_lower) and \
           ("доставк" in key_lower or "delivery" in key_lower):
            markup_str = system_vars[key]
            return parse_markup_value(markup_str)
    
    return None, None


def calculate_final_price(
    order_price_kg: Union[float, str, None],
    system_vars: Dict[str, str],
    supplier_name: Optional[str] = None,
) -> str:
    """Рассчитывает финальную цену с учетом наценок из системных переменных.
    
    ВАЖНО: Для товаров поставщика "ООО КИТ" цена уже финальная в БД, наценки не применяются."""
    try:
        if order_price_kg is None:
            return "Цена по запросу"
        
        if isinstance(order_price_kg, str):
            price_str = order_price_kg.strip()
            if not price_str or price_str == "Не указано":
                return "Цена по запросу"
            try:
                order_price_kg = float(price_str)
            except (ValueError, TypeError):
                return "Цена по запросу"
        
        order_price_kg_float = float(order_price_kg)
        
        if order_price_kg_float == 0:
            return "Цена по запросу"
        
        if supplier_name:
            supplier_normalized = supplier_name.upper().strip()
            if "КИТ" in supplier_normalized and ("ООО" in supplier_normalized or supplier_normalized.startswith("КИТ")):
                final_price_rounded = round(order_price_kg_float, 2)
                logger.debug(
                    f"Price for supplier '{supplier_name}' (КИТ) is already final: {final_price_rounded}"
                )
                return f"{final_price_rounded:.2f}"
        
        markup_percentage, markup_absolute = get_markup_from_system_vars(
            order_price_kg_float, system_vars
        )
        
        delivery_percentage, delivery_absolute = get_delivery_markup(system_vars)
        
        final_price = order_price_kg_float
        
        if markup_percentage is not None:
            final_price = final_price * (1 + markup_percentage / 100)
        
        if markup_absolute is not None:
            final_price = final_price + markup_absolute
        
        if delivery_percentage is not None:
            final_price = final_price * (1 + delivery_percentage / 100)
        
        if delivery_absolute is not None:
            final_price = final_price + delivery_absolute
        
        final_price_rounded = round(final_price, 2)
        
        return f"{final_price_rounded:.2f}"
        
    except Exception as e:
        logger.error(f"Error calculating final price for {order_price_kg}: {e}", exc_info=True)
        return "Цена по запросу"

