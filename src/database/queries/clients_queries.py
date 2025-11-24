"""SQL запросы для работы с клиентами."""

from typing import Optional

from src.models.entities import Client
from src.utils import get_supabase_client


async def get_client_by_phone(phone: str) -> Optional[Client]:
    """Получает профиль клиента по номеру телефона.

    Args:
        phone: Номер телефона клиента

    Returns:
        Модель Client или None если не найден
    """
    try:
        supabase = await get_supabase_client()
        result = await supabase.table("clients").select("*").eq("phone", phone).execute()
        if result.data and len(result.data) > 0:
            return Client(**result.data[0])
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
    if profile.name:
        profile_parts.append(f"Имя: {profile.name}")
    if profile.phone:
        profile_parts.append(f"Телефон: {profile.phone}")
    if profile.city:
        profile_parts.append(f"Город: {profile.city}")
    if profile.business_area:
        profile_parts.append(f"Бизнес-область: {profile.business_area}")
    if profile.org_name:
        profile_parts.append(f"Организация: {profile.org_name}")
    if profile.is_it_friend:
        profile_parts.append("Статус: Друг компании")
    if profile.mode:
        profile_parts.append(f"Режим: {profile.mode}")
    if profile.UTC is not None:
        profile_parts.append(f"Часовой пояс: UTC{profile.UTC}")

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
    
    return bool(profile.is_it_friend)


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
        return True  # По умолчанию разрешено
    
    send_message = profile.get("send_message")
    return bool(send_message) if send_message is not None else True

