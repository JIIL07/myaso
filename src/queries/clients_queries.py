"""SQL запросы для работы с клиентами."""

from typing import Any, Dict, Optional

from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout


async def get_client_by_phone(phone: str) -> Optional[Dict[str, Any]]:
    """Получает профиль клиента по номеру телефона.

    Args:
        phone: Номер телефона клиента

    Returns:
        Словарь с данными клиента или None если не найден
        
    Raises:
        RuntimeError: Если произошла ошибка при выполнении запроса или timeout
    """
    try:
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table("clients").select("*").eq("phone", phone).execute(),
            operation_name=f"get_client_by_phone({phone})",
        )
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


async def get_client_send_message(phone: str) -> bool:
    """Получает значение send_message для клиента.

    Args:
        phone: Номер телефона клиента

    Returns:
        True если отправка сообщений разрешена, False если запрещена.
        По умолчанию возвращает True если клиент не найден или поле не установлено.
    """
    profile = await get_client_by_phone(phone)
    if not profile:
        return True
    
    send_message = profile.get("send_message")
    return bool(send_message) if send_message is not None else True


async def get_client_is_friend(phone: str) -> bool:
    """Проверяет, является ли клиент другом компании.

    Args:
        phone: Номер телефона клиента

    Returns:
        True если клиент является другом компании, False иначе.
        По умолчанию возвращает False если клиент не найден или поле не установлено.
    """
    profile = await get_client_by_phone(phone)
    if not profile:
        return False
    
    is_it_friend = profile.get("is_it_friend")
    return bool(is_it_friend) if is_it_friend is not None else False


async def get_client_style(phone: str) -> Optional[str]:
    """Получает стиль общения клиента из базы данных.

    Args:
        phone: Номер телефона клиента

    Returns:
        Стиль общения клиента ('Эдуард', 'Полина', 'Маша') или None если не установлен.
    """
    profile = await get_client_by_phone(phone)
    if not profile:
        return None
    
    style = profile.get("style")
    if style and isinstance(style, str) and style.strip():
        return style.strip()
    return None
