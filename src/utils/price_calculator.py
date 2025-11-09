"""Утилиты для расчета финальных цен товаров."""

import logging
import re
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


def parse_markup_percentage(markup_str: Optional[str]) -> float:
    """Парсит процент наценки из строки формата "10%".
    
    Args:
        markup_str: Строка с процентом наценки (например, "10%")
    
    Returns:
        Число процентов (например, 10.0)
        
    Raises:
        ValueError: Если не удалось распарсить процент
    """
    if not markup_str:
        raise ValueError("Markup string is empty")
    
    markup_clean = markup_str.strip()
    match = re.search(r'(\d+\.?\d*)', markup_clean)
    
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Could not parse markup percentage from: {markup_str}")


def get_markup_from_system_vars(
    order_price_kg: float,
    system_vars: Dict[str, str]
) -> float:
    """Получает процент наценки из системных переменных на основе цены.
    
    Args:
        order_price_kg: Базовая цена за кг
        system_vars: Словарь системных переменных (topic -> value)
    
    Returns:
        Процент наценки (например, 10.0 для 10%)
        
    Raises:
        ValueError: Если не найдена подходящая наценка в system_vars
    """
    if order_price_kg < 100:
        topic = "Наценка на кг/руб (<100 руб)"
    else:
        topic = "Наценка на кг/руб (>100 руб)"
    
    markup_str = system_vars.get(topic)
    
    if not markup_str:
        for key in system_vars.keys():
            if "наценка" in key.lower() and "<100" in key.lower() and order_price_kg < 100:
                markup_str = system_vars[key]
                break
            elif "наценка" in key.lower() and ">100" in key.lower() and order_price_kg >= 100:
                markup_str = system_vars[key]
                break
        
        if not markup_str:
            logger.warning(
                f"Markup not found for price {order_price_kg}. "
                f"Available keys: {list(system_vars.keys())}"
            )
            return 0.0
    
    try:
        return parse_markup_percentage(markup_str)
    except ValueError as e:
        logger.error(f"Error parsing markup '{markup_str}': {e}")
        return 0.0


def calculate_final_price(
    order_price_kg: Union[float, str, None],
    system_vars: Dict[str, str],
) -> str:
    """Рассчитывает финальную цену с учетом наценки.
    
    Args:
        order_price_kg: Базовая цена за кг из БД (может быть None, 0, float, или строка)
        system_vars: Словарь системных переменных (topic -> value)
    
    Returns:
        Финальная цена как строка:
        - "Цена по запросу" если order_price_kg == 0, None, или пустая строка
        - Иначе: строка с ценой, округленной до 2 знаков (например, "385.00")
    """
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
        
        markup_percent = get_markup_from_system_vars(order_price_kg_float, system_vars)
        final_price = order_price_kg_float * (1 + markup_percent / 100)
        final_price_rounded = round(final_price, 2)
        
        return f"{final_price_rounded:.2f}"
        
    except Exception as e:
        logger.error(f"Error calculating final price for {order_price_kg}: {e}", exc_info=True)
        return "Цена по запросу"

