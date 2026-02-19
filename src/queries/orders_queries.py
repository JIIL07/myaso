from src.entities.product import Product


async def get_client_orders(client_phone: str) -> list[Product]:
    return []


async def get_last_order(client_phone: str) -> Product | None:
    return None
