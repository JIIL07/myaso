from typing import Any, Optional

from src.services.database.database import get_pool


async def get_client_by_phone(phone: str) -> Optional[dict[str, Any]]:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT *
                FROM myaso.clients
                WHERE phone = $1
                LIMIT 1
                """,
                phone,
            )
        if row:
            return dict(row)
        return None
    except Exception as e:
        raise RuntimeError("Error getting client: %s" % e) from e


async def get_client_profile_text(phone: str) -> str:
    profile = await get_client_by_phone(phone)
    if not profile:
        return "Профиль клиента не найден в базе данных."

    profile_parts = []
    if profile.get("name"):
        profile_parts.append("Имя: %s" % profile["name"])
    if profile.get("phone"):
        profile_parts.append("Телефон: %s" % profile["phone"])
    if profile.get("city"):
        profile_parts.append("Город: %s" % profile["city"])
    if profile.get("business_area"):
        profile_parts.append("Бизнес-область: %s" % profile["business_area"])
    if profile.get("org_name"):
        profile_parts.append("Организация: %s" % profile["org_name"])
    if profile.get("is_it_friend"):
        profile_parts.append("Статус: Друг компании")
    if profile.get("mode"):
        profile_parts.append("Режим: %s" % profile["mode"])
    if profile.get("UTC") is not None:
        profile_parts.append("Часовой пояс: UTC%s" % profile["UTC"])

    return (
        "\n".join(profile_parts)
        if profile_parts
        else "Профиль найден, но данные отсутствуют."
    )


async def get_client_send_message(phone: str) -> bool:
    profile = await get_client_by_phone(phone)
    if not profile:
        return True

    send_message = profile.get("send_message")
    return bool(send_message) if send_message is not None else True


async def get_client_is_friend(phone: str) -> bool:
    profile = await get_client_by_phone(phone)
    if not profile:
        return False

    is_it_friend = profile.get("is_it_friend")
    return bool(is_it_friend) if is_it_friend is not None else False


async def get_client_style(phone: str) -> Optional[str]:
    profile = await get_client_by_phone(phone)
    if not profile:
        return None

    style = profile.get("style")
    if style and isinstance(style, str) and style.strip():
        return style.strip()
    return None
