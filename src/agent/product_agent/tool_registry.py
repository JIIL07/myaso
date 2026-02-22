from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.tools.execute_sql import execute_sql_query
from src.tools.generate_sql import generate_sql_from_text
from src.tools.get_client_orders import get_client_orders
from src.tools.get_client_profile import get_client_profile
from src.tools.get_product_by_title import get_product_by_title
from src.tools.get_random_products import get_random_products
from src.tools.get_schema import get_database_schema
from src.tools.send_pricelist import send_pricelist
from src.tools.set_photo_requirement import set_photo_requirement
from src.tools.show_product_photos import show_product_photos
from src.tools.vector_search import vector_search


@dataclass(frozen=True)
class ToolRegistryFlags:
    enable_sql_tools: bool = True
    enable_media_tools: bool = True
    enable_context_tools: bool = True


def get_core_tools() -> list[Any]:
    return [
        get_client_profile,
        get_client_orders,
        vector_search,
        get_product_by_title,
        get_random_products,
        get_database_schema,
    ]


def get_sql_tools() -> list[Any]:
    return [generate_sql_from_text, execute_sql_query, get_database_schema]


def get_media_tools() -> list[Any]:
    return [show_product_photos, send_pricelist]


def get_context_tools() -> list[Any]:
    return [set_photo_requirement]


def build_agent_tools(base_tools: list[Any], flags: ToolRegistryFlags | None = None) -> list[Any]:
    flags = flags or ToolRegistryFlags()
    all_tools = list(base_tools)

    if flags.enable_sql_tools:
        all_tools.extend(get_sql_tools())
    if flags.enable_media_tools:
        all_tools.extend(get_media_tools())
    if flags.enable_context_tools:
        all_tools.extend(get_context_tools())

    return all_tools
