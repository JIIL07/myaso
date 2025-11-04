"""LangChain Tools для замены Mirascope tools.

Содержит функции с декоратором @tool для использования в LangChain агентах.
"""

from __future__ import annotations

from typing import List
import requests
from langchain_core.tools import tool
from supabase import create_client, Client, ClientOptions

from src.config.settings import settings
from src.utils.langchain_retrievers import SupabaseVectorRetriever
from src.services.profile_service import ProfileService


# Инициализация Supabase client для синхронных операций
supabase_client = create_client(
    settings.supabase.supabase_url,
    settings.supabase.supabase_service_key,
    options=ClientOptions(schema="myaso"),
)


@tool
async def enhance_user_product_query(query: str) -> str:
    """Ищет товары в базе данных по семантическому запросу пользователя.

    Использует векторный поиск (embeddings) для нахождения релевантных товаров.
    Возвращает отформатированный список товаров с их характеристиками.

    Используй этот инструмент когда:
    - Пользователь запрашивает товары/ассортимент
    - Запрос содержит критерии поиска (тип мяса, часть туши, формат, вес, упаковка)
    - Пользователь спрашивает "Что у вас есть?", "Покажи мясо", "Какие стейки есть?"
    - Пользователь просит рекомендацию с критериями ("Что подходит для гриля?")

    НЕ используй если:
    - Запрос содержит только подтверждение/отказ ("Да", "Нет", "Ок")
    - Обсуждаются сервисные темы (доставка, оплата, расписание)
    - Пользователь уточняет детали уже известного товара
    - Запрос слишком общий без критериев

    Args:
        query: Текстовый запрос пользователя о товарах/ассортименте

    Returns:
        Строка с отформатированным списком найденных товаров и их характеристиками
    """
    retriever = SupabaseVectorRetriever()
    documents = await retriever.get_relevant_documents(query, k=10)

    if not documents:
        return "Товары по вашему запросу не найдены."

    products_list = []
    for doc in documents:
        metadata = doc.metadata
        product_info = [
            f"Название: {metadata.get('title', 'Не указано')}",
            f"Поставщик: {metadata.get('supplier_name', 'Не указано')}",
            f"Регион: {metadata.get('from_region', 'Не указано')}",
            f"Цена за кг: {metadata.get('order_price_kg', 'Не указано')}",
            f"Минимальный заказ (кг): {metadata.get('min_order_weight_kg', 'Не указано')}",
            f"Охлаждённый/Замороженный: {metadata.get('cooled_or_frozen', 'Не указано')}",
            f"Полуфабрикат: {'Да' if metadata.get('ready_made') else 'Нет'}",
            f"Тип упаковки: {metadata.get('package_type', 'Не указано')}",
            f"Скидка: {metadata.get('discount', 'Не указано')}",
        ]
        products_list.append("\n".join([info for info in product_info if "Не указано" not in info]))

    result = "\n\n---\n\n".join(products_list)
    return f"Найдено товаров: {len(documents)}\n\n{result}"


@tool
async def show_product_photos(product_titles: List[str], phone: str) -> str:
    """Отправляет фотографии товаров пользователю через WhatsApp API.

    Находит товары по их названиям в базе данных и отправляет фотографии
    на указанный номер телефона через WhatsApp API.

    Используй этот инструмент когда:
    - Пользователь просит показать фото товаров
    - Нужно визуально представить товары пользователю
    - После поиска товаров, если пользователь хочет увидеть фото

    Args:
        product_titles: Список названий товаров, фото которых нужно отправить
        phone: Номер телефона пользователя в формате WhatsApp

    Returns:
        Строка с статусом отправки фотографий (какие отправлены, каких нет)
    """
    has_photo = []
    no_photo = []
    not_found = []

    for title in product_titles:
        response = (
            supabase_client.table("products")
            .select("*")
            .eq("title", title)
            .execute()
        )

        if not response.data or len(response.data) == 0:
            not_found.append(title)
            continue

        found_with_photo = False
        for product in response.data:
            photo_url = product.get("photo")
            
            if photo_url:
                try:
                    requests.post(
                        url="http://51.250.42.45:2026/sendImage",
                        json={
                            "recipient": phone,
                            "image_url": photo_url,
                            "caption": title,
                        },
                        timeout=10,
                    )
                    found_with_photo = True
                except Exception as e:
                    print(f"Ошибка отправки фото для {title}: {e}")

        if found_with_photo:
            has_photo.append(title)
        else:
            no_photo.append(title)

    result_parts = []
    if has_photo:
        result_parts.append(f"Фотографии следующих товаров отправлены: {', '.join(has_photo)}")
    if no_photo:
        result_parts.append(f"Нет фотографий следующих товаров: {', '.join(no_photo)}")
    if not_found:
        result_parts.append(f"Товары не найдены: {', '.join(not_found)}")

    return "\n".join(result_parts) if result_parts else "Нет товаров для отправки фотографий."


@tool
async def get_client_profile(phone: str) -> str:
    """Получает профиль клиента из базы данных.

    Возвращает информацию о клиенте: имя, контакты, город, бизнес-данные и т.д.

    Используй этот инструмент когда:
    - Нужна информация о клиенте для персонализации ответов
    - Нужно узнать город, бизнес-область или другие данные клиента
    - Нужно адаптировать предложения под профиль клиента

    Args:
        phone: Номер телефона клиента

    Returns:
        Строка с отформатированной информацией о профиле клиента или сообщение об отсутствии данных
    """
    profile_service = await ProfileService()
    profile = await profile_service.get_profile(client_phone=phone)

    if not profile or (isinstance(profile, list) and len(profile) == 0):
        return "Профиль клиента не найден в базе данных."

    profile_parts = []
    if isinstance(profile, dict):
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

    result = "\n".join(profile_parts) if profile_parts else "Профиль найден, но данные отсутствуют."
    return result
