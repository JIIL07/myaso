"""Утилиты для расчета финальных цен товаров."""

import logging
import re
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_markup_value(markup_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Парсит значение наценки из строки.
    
    Определяет, является ли значение процентом (с символом %) или абсолютной суммой в рублях.
    
    Args:
        markup_str: Строка с наценкой (например, "10%" или "10")
    
    Returns:
        Кортеж (percentage, absolute):
        - Если значение содержит "%": (процент, None) - например, (10.0, None) для "10%"
        - Если значение просто число: (None, абсолютная_сумма) - например, (None, 10.0) для "10"
        - Если не удалось распарсить: (None, None)
    """
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
    """Получает наценку из системных переменных на основе цены.
    
    Args:
        order_price_kg: Базовая цена за кг
        system_vars: Словарь системных переменных (topic -> value)
    
    Returns:
        Кортеж (percentage, absolute):
        - percentage: Процент наценки (если значение с %), иначе None
        - absolute: Абсолютная наценка в рублях (если значение просто число), иначе None
    """
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
    """Получает дополнительную наценку из системных переменных.
    
    Ищет переменные, содержащие ключевые слова: "наценк" и "доставк".
    
    Args:
        system_vars: Словарь системных переменных (topic -> value)
    
    Returns:
        Кортеж (percentage, absolute):
        - percentage: Процент дополнительной наценки (если значение с %), иначе None
        - absolute: Абсолютная дополнительная наценка в рублях (если значение просто число), иначе None
    """
    for key in system_vars.keys():
        key_lower = key.lower()
        if ("наценк" in key_lower or "markup" in key_lower) and \
           ("доставк" in key_lower or "delivery" in key_lower):
            markup_str = system_vars[key]
            logger.debug(f"Found delivery markup in key '{key}': {markup_str}")
            return parse_markup_value(markup_str)
    
    logger.debug("Delivery markup not found in system_vars")
    return None, None


def calculate_final_price(
    order_price_kg: Union[float, str, None],
    system_vars: Dict[str, str],
    supplier_name: Optional[str] = None,
) -> str:
    """Рассчитывает финальную цену с учетом наценок из системных переменных.
    
    Args:
        order_price_kg: Базовая цена за кг из БД (может быть None, 0, float, или строка)
        system_vars: Словарь системных переменных (topic -> value)
        supplier_name: Название поставщика (опционально). Если поставщик "ООО "КИТ"", 
                       финальная цена не рассчитывается
    
    Returns:
        Финальная цена как строка:
        - "Цена по запросу" если order_price_kg == 0, None, или пустая строка
        - "Цена по запросу" если поставщик "ООО "КИТ""
        - Иначе: строка с ценой, округленной до 2 знаков (например, "385.00")
    """
    try:
        # Если поставщик "ООО "КИТ"", не рассчитываем финальную цену
        if supplier_name:
            supplier_normalized = supplier_name.upper().strip()
            # Проверяем разные варианты написания: "ООО КИТ", "ООО"КИТ"", "КИТ"
            if "КИТ" in supplier_normalized and ("ООО" in supplier_normalized or supplier_normalized.startswith("КИТ")):
                logger.debug(f"Поставщик {supplier_name} - ООО КИТ, финальная цена не рассчитывается")
                return "Цена по запросу"
        
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
        
        markup_percentage, markup_absolute = get_markup_from_system_vars(
            order_price_kg_float, system_vars
        )
        
        delivery_percentage, delivery_absolute = get_delivery_markup(system_vars)
        
        final_price = order_price_kg_float
        
        if markup_percentage is not None:
            final_price = final_price * (1 + markup_percentage / 100)
            logger.debug(
                f"Applied percentage markup {markup_percentage}%: "
                f"{order_price_kg_float} -> {final_price}"
            )
        
        if markup_absolute is not None:
            final_price = final_price + markup_absolute
            logger.debug(
                f"Applied absolute markup {markup_absolute} руб: "
                f"price -> {final_price}"
            )
        
        if delivery_percentage is not None:
            final_price = final_price * (1 + delivery_percentage / 100)
            logger.debug(
                f"Applied delivery percentage markup {delivery_percentage}%: "
                f"price -> {final_price}"
            )
        
        if delivery_absolute is not None:
            final_price = final_price + delivery_absolute
            logger.debug(
                f"Applied delivery absolute markup {delivery_absolute} руб: "
                f"price -> {final_price}"
            )
        
        final_price_rounded = round(final_price, 2)
        
        return f"{final_price_rounded:.2f}"
        
    except Exception as e:
        logger.error(f"Error calculating final price for {order_price_kg}: {e}", exc_info=True)
        return "Цена по запросу"

