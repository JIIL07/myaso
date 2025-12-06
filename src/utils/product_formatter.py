"""Утилиты для форматирования товаров."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.utils.field_normalizer import normalize_field_value
from src.utils.price_calculator import calculate_final_price
from src.utils.prompts import get_all_system_values


async def format_product(
    product: Dict[str, Any],
    system_vars: Optional[Dict[str, str]] = None,
) -> str:
    """Форматирует один товар в читаемый текст.

    Args:
        product: Словарь с данными товара (должен содержать title, supplier_name, order_price_kg, from_region)
        system_vars: Словарь системных переменных для расчета цены (если None, загружается автоматически)

    Returns:
        Отформатированная строка с информацией о товаре
    """
    if system_vars is None:
        system_vars = await get_all_system_values()

    title = product.get("title", "Не указано")
    supplier = normalize_field_value(product.get("supplier_name"), "text")
    order_price = product.get("order_price_kg")
    region = normalize_field_value(product.get("from_region"), "text")

    final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)

    product_lines = [f"📦 {title}"]
    product_lines.append(f"   Поставщик: {supplier}")

    if final_price != "Цена по запросу":
        product_lines.append(f"   Цена: {final_price}₽/кг")
    else:
        product_lines.append(f"   Цена: {final_price}")

    product_lines.append(f"   Регион: {region}")

    return "\n".join(product_lines)


async def format_products_list(
    products: List[Dict[str, Any]],
    system_vars: Optional[Dict[str, str]] = None,
) -> tuple[str, List[int]]:
    """Форматирует список товаров и извлекает их ID.

    Args:
        products: Список словарей с данными товаров
        system_vars: Словарь системных переменных для расчета цены (если None, загружается автоматически)

    Returns:
        Кортеж (отформатированный текст, список ID товаров)
    """
    if system_vars is None:
        system_vars = await get_all_system_values()

    products_list = []
    product_ids = []

    for product in products:
        product_id = product.get("id")
        if product_id:
            product_ids.append(product_id)

        formatted_product = await format_product(product, system_vars)
        products_list.append(formatted_product)

    result_text = "\n\n".join(products_list)
    return result_text, product_ids


def create_product_ids_section(product_ids: List[int]) -> str:
    """Создает секцию [PRODUCT_IDS] для ответа инструмента.

    Args:
        product_ids: Список ID товаров

    Returns:
        Строка с секцией [PRODUCT_IDS] или пустая строка если список пуст
    """
    if not product_ids:
        return ""

    ids_json = json.dumps({"product_ids": product_ids})
    return f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]"

