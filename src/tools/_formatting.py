"""Shared formatting helpers for product search tools."""

from __future__ import annotations

import logging
from typing import Any
from src.services.ai.prompt import get_all_system_values
from src.toolkit import filter_products_with_photo, format_products_list

logger = logging.getLogger(__name__)


async def format_and_return_products(
    products: list[dict[str, Any]],
    *,
    require_photo: bool = False,
    limit: int | None = None,
    empty_message: str = "Товары не найдены.",
    no_photo_message: str = "Товары с фотографиями не найдены.",
) -> tuple[str, list[int]]:
    """Filter by photo (if needed), format, and return (text, product_ids)."""
    if not products:
        return empty_message, []

    if require_photo:
        products = filter_products_with_photo(products)
        if not products:
            return no_photo_message, []

    if limit is not None:
        products = products[:limit]

    system_vars = await get_all_system_values()
    result_text, product_ids = await format_products_list(products, system_vars)

    return f"Найдено товаров: {len(products)}\n\n{result_text}", product_ids
