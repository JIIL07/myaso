from __future__ import annotations

import re
from typing import Any

import asyncpg

def remove_markdown_symbols(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"#{1,6}\s+(.+?)$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    text = re.sub(r"^[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    return text.strip()


def records_to_json(records: list[asyncpg.Record]) -> list[dict[str, Any]]:
    return [dict(record) for record in records]


def normalize_field_value_sync(value, field_type: str = "text") -> str:
    from src.constants import DEFAULT_FIELD_VALUE
    
    empty_values = ["не указано", "null", "none", ""]

    if value is None:
        return DEFAULT_FIELD_VALUE

    if field_type == "text":
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in empty_values:
                return DEFAULT_FIELD_VALUE
            return value_str
        elif isinstance(value, (int, float)) and value == 0:
            return DEFAULT_FIELD_VALUE
        return str(value).strip() if str(value).strip() else DEFAULT_FIELD_VALUE
    else:
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str or value_str.lower() in empty_values:
                return DEFAULT_FIELD_VALUE
            try:
                num_value = float(value_str)
                if num_value == 0:
                    return DEFAULT_FIELD_VALUE
                return str(int(num_value)) if num_value.is_integer() else str(num_value)
            except (ValueError, TypeError):
                return DEFAULT_FIELD_VALUE
        elif isinstance(value, (int, float)):
            if value == 0:
                return DEFAULT_FIELD_VALUE
            return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
        return DEFAULT_FIELD_VALUE


async def format_product(product: dict[str, Any], system_vars: dict[str, str]) -> str:
    from src.utils.prices.price_calculator import calculate_final_price
    
    title = product.get("title", "Не указано")
    supplier = normalize_field_value_sync(product.get("supplier_name"), "text")
    order_price = product.get("order_price_kg")
    region = normalize_field_value_sync(product.get("from_region"), "text")

    final_price = calculate_final_price(order_price, system_vars, supplier_name=supplier)

    product_lines = [f"📦 {title}"]
    product_lines.append(f"   Поставщик: {supplier}")

    if final_price != "Цена по запросу":
        product_lines.append(f"   Цена: {final_price}₽/кг")
    else:
        product_lines.append(f"   Цена: {final_price}")

    product_lines.append(f"   Регион: {region}")

    return "\n".join(product_lines)


async def format_products_list(products: list[dict[str, Any]], system_vars: dict[str, str]) -> tuple[str, list[int]]:
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


def has_photo(product: dict[str, Any]) -> bool:
    photo = product.get("photo")
    return bool(photo and str(photo).strip())


def filter_products_by_photo(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [p for p in products if has_photo(p)]

