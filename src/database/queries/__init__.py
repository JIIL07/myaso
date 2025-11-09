"""SQL queries и операции с базой данных."""

from .products_queries import (
    get_random_products,
    get_products_by_sql_conditions,
    get_product_by_title,
)
from .clients_queries import (
    get_client_by_phone,
    get_client_profile_text,
)
from .orders_queries import (
    get_client_orders,
    get_last_order,
)

__all__ = [
    "get_random_products",
    "get_products_by_sql_conditions",
    "get_product_by_title",
    "get_client_by_phone",
    "get_client_profile_text",
    "get_client_orders",
    "get_last_order",
]

