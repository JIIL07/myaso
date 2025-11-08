"""SQL запросы для работы с клиентами."""

from typing import Dict, Any, Optional
from src.utils import get_supabase_client


async def get_client_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """Получает профиль клиента по номеру телефона.

    Args:
        phone: Номер телефона клиента

    Returns:
        Словарь с данными клиента или None если не найден
    """
    try:
        supabase = await get_supabase_client()
        result = await supabase.table("clients").select("*").eq("phone", phone).execute()
        if result.data and len(result.data) > 0:
            return result.data[0]
        return None
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении клиента: {e}") from e


async def get_client_profile_text(phone: str) -> str:
    """Получает текстовое представление профиля клиента.

    Args:
        phone: Номер телефона клиента

    Returns:
        Строка с отформатированной информацией о профиле клиента
    """
    profile = await get_client_by_phone(phone)
    if not profile:
        return "Профиль клиента не найден в базе данных."

    profile_parts = []
    if profile.get("name"):
        profile_parts.append(f"Имя: {profile['name']}")
    if profile.get("phone"):
        profile_parts.append(f"Телефон: {profile['phone']}")
    if profile.get("city"):
        profile_parts.append(f"Город: {profile['city']}")
    if profile.get("business_area"):
        profile_parts.append(f"Бизнес-область: {profile['business_area']}")
    if profile.get("org_name"):
        profile_parts.append(f"Организация: {profile['org_name']}")
    if profile.get("is_it_friend"):
        profile_parts.append("Статус: Друг компании")
    if profile.get("mode"):
        profile_parts.append(f"Режим: {profile['mode']}")
    if profile.get("UTC") is not None:
        profile_parts.append(f"Часовой пояс: UTC{profile['UTC']}")

    return (
        "\n".join(profile_parts)
        if profile_parts
        else "Профиль найден, но данные отсутствуют."
    )


async def get_client_is_friend(phone: str) -> bool:
    """Получает флаг дружбы клиента.

    Args:
        phone: Номер телефона клиента

    Returns:
        True если клиент является другом (it_is_friend=TRUE), False в противном случае
    """
    profile = await get_client_by_phone(phone)
    if not profile:
        return False
    
    is_friend = profile.get("is_it_friend")
    return bool(is_friend)

