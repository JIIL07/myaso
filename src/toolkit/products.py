from __future__ import annotations

from typing import Any

from src.constants import DEFAULT_FIELD_VALUE
from src.utils.prices.price_calculator import calculate_final_price


def normalize_field_value(value: Any, field_type: str = "text") -> str:
    empty_values = ["не указано", "null", "none", ""]
    if value is None:
        return DEFAULT_FIELD_VALUE

    if field_type == "text":
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in empty_values:
                return DEFAULT_FIELD_VALUE
            return value_str
        if isinstance(value, (int, float)) and value == 0:
            return DEFAULT_FIELD_VALUE
        value_str = str(value).strip()
        return value_str if value_str else DEFAULT_FIELD_VALUE

    if isinstance(value, str):
        value_str = value.strip()
        if not value_str or value_str.lower() in empty_values:
            return DEFAULT_FIELD_VALUE
        try:
            num_value = float(value_str)
        except (ValueError, TypeError):
            return DEFAULT_FIELD_VALUE
        if num_value == 0:
            return DEFAULT_FIELD_VALUE
        return str(int(num_value)) if num_value.is_integer() else str(num_value)

    if isinstance(value, (int, float)):
        if value == 0:
            return DEFAULT_FIELD_VALUE
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)

    return DEFAULT_FIELD_VALUE


def has_product_photo(product: dict[str, Any]) -> bool:
    photo = product.get("photo")
    return bool(photo and str(photo).strip())


def filter_products_with_photo(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [product for product in products if has_product_photo(product)]


async def format_product(product: dict[str, Any], system_vars: dict[str, str]) -> str:
    title = product.get("title", "Не указано")
    supplier = normalize_field_value(product.get("supplier_name"), "text")
    order_price = product.get("order_price_kg")
    region = normalize_field_value(product.get("from_region"), "text")

    final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)
    product_lines = [
        f"📦 {title}",
        f"   Поставщик: {supplier}",
        f"   Цена: {final_price}₽/кг" if final_price != "Цена по запросу" else f"   Цена: {final_price}",
        f"   Регион: {region}",
    ]
    return "\n".join(product_lines)


async def format_products_list(
    products: list[dict[str, Any]], system_vars: dict[str, str]
) -> tuple[str, list[int]]:
    products_list: list[str] = []
    product_ids: list[int] = []

    for product in products:
        product_id = product.get("id")
        if product_id:
            product_ids.append(product_id)
        products_list.append(await format_product(product, system_vars))

    return "\n\n".join(products_list), product_ids
