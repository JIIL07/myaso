"""LangChain Tools.

Содержит функции с декоратором @tool для использования в LangChain агентах.
"""

from __future__ import annotations

from typing import List, Any
import logging
import os
import httpx
import asyncpg
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from supabase import create_client, ClientOptions, acreate_client, AClient, AsyncClientOptions

from src.config.settings import settings
from src.config.langchain_settings import LangChainSettings
from src.utils.langchain_retrievers import SupabaseVectorRetriever
from src.utils import records_to_json
from src.utils.prompts import get_prompt

logger = logging.getLogger(__name__)
langchain_settings = LangChainSettings()


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

    try:
        documents = await retriever.get_relevant_documents(query, k=15)
    except Exception as e:
        logger.warning(f"Ошибка при поиске по запросу '{query}': {e}")
        return "Товары по вашему запросу не найдены."
    
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
    if not product_titles:
        return "Нет товаров для отправки фотографий."
    
    has_photo = []
    no_photo = []
    not_found = []

    supabase: AClient = await acreate_client(
        settings.supabase.supabase_url,
        settings.supabase.supabase_service_key,
        options=AsyncClientOptions(schema="myaso")
    )

    for title in product_titles:
        try:
            result = await supabase.table("products").select("*").eq("title", title).execute()
            
            if not result.data or len(result.data) == 0:
                not_found.append(title)
                continue

            found_with_photo = False
            for product in result.data:
                photo_url = product.get("photo")
                
                if photo_url:
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            response = await client.post(
                                url=settings.whatsapp.send_image_url,
                                json={
                                    "recipient": phone,
                                    "image_url": photo_url,
                                    "caption": title,
                                },
                            )
                            response.raise_for_status()
                        found_with_photo = True
                        logger.info(f"[show_product_photos] Фото товара '{title}' отправлено для {phone}")
                    except Exception as e:
                        logger.warning(f"[show_product_photos] Ошибка отправки фото для {title}: {e}")

            if found_with_photo:
                has_photo.append(title)
            else:
                no_photo.append(title)
        except Exception as e:
            logger.error(f"[show_product_photos] Ошибка при поиске товара '{title}': {e}")
            not_found.append(title)

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
    supabase: AClient = await acreate_client(
        settings.supabase.supabase_url,
        settings.supabase.supabase_service_key,
        options=AsyncClientOptions(schema="myaso")
    )
    
    result = await supabase.table('clients').select('*').eq('phone', phone).execute()
    profile = result.data[0] if result.data and len(result.data) > 0 else None

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

    result_text = "\n".join(profile_parts) if profile_parts else "Профиль найден, но данные отсутствуют."
    return result_text


@tool
async def get_client_orders(phone: str) -> str:
    """Получает заказы клиента из базы данных.

    Возвращает список заказов клиента с информацией о товарах, ценах, датах доставки.

    Args:
        phone: Номер телефона клиента

    Returns:
        Строка с отформатированным списком заказов или сообщение об отсутствии заказов
    """
    supabase: AClient = await acreate_client(
        settings.supabase.supabase_url,
        settings.supabase.supabase_service_key,
        options=AsyncClientOptions(schema="myaso")
    )
    
    result = await supabase.table("orders").select("*").eq("client_phone", phone).order("created_at", desc=True).execute()
    
    if not result.data or len(result.data) == 0:
        return "Заказы не найдены."
    
    orders_list = []
    for order in result.data:
        order_info = [
            f"Товар: {order.get('title', 'Не указано')}",
            f"Дата: {order.get('created_at', 'Не указано')}",
            f"Вес (кг): {order.get('weight_kg', 'Не указано')}",
            f"Цена: {order.get('price_out', 'Не указано')}",
            f"Пункт назначения: {order.get('destination', 'Не указано')}",
        ]
        orders_list.append("\n".join([info for info in order_info if "Не указано" not in info]))
    
    result_text = "\n\n---\n\n".join(orders_list)
    return f"Найдено заказов: {len(result.data)}\n\n{result_text}"


@tool
async def get_random_products(limit: int = 10) -> str:
    """Получает случайные товары из ассортимента.

    Используется как fallback когда поиск по запросу не дал результатов.
    Возвращает случайный набор товаров из базы данных.

    Args:
        limit: Количество товаров для возврата (по умолчанию 10)

    Returns:
        Строка с отформатированным списком случайных товаров
    """
    db_dsn = os.getenv("POSTGRES_DSN") or None
    if not db_dsn:
        return "Не настроено подключение к базе данных."
    
    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        result = await conn.fetch(
            """
            SELECT 
                id,
                title,
                supplier_name,
                from_region,
                photo,
                order_price_kg,
                min_order_weight_kg,
                cooled_or_frozen,
                ready_made,
                package_type,
                discount
            FROM myaso.products
            ORDER BY RANDOM()
            LIMIT $1
            """,
            limit
        )
        json_result = records_to_json(result)
        
        if not json_result:
            return "Товары не найдены."
        
        products_list = []
        for product in json_result:
            product_info = [
                f"Название: {product.get('title', 'Не указано')}",
                f"Поставщик: {product.get('supplier_name', 'Не указано')}",
                f"Регион: {product.get('from_region', 'Не указано')}",
                f"Цена за кг: {product.get('order_price_kg', 'Не указано')}",
                f"Минимальный заказ (кг): {product.get('min_order_weight_kg', 'Не указано')}",
            ]
            products_list.append("\n".join([info for info in product_info if "Не указано" not in info]))
        
        result_text = "\n\n---\n\n".join(products_list)
        return f"Найдено товаров: {len(json_result)}\n\n{result_text}"
        
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}"
    finally:
        if conn:
            await conn.close()


@tool
async def generate_sql_from_text(
    text_conditions: str,
    previous_sql: str = None,
    error_message: str = None,
    attempt_number: int = 1
) -> str:
    """Генерирует SQL WHERE условия из текстового описания используя LLM.
    
    Преобразует текстовое описание условий на русском языке в SQL WHERE условия.
    
    Используй этот инструмент когда:
    - Нужно преобразовать текстовое описание условий в SQL
    - Пользователь описывает условия для поиска товаров на русском языке
    - Требуется сгенерировать SQL запрос для фильтрации товаров
    
    Args:
        text_conditions: Текстовое описание условий на русском языке (например, "товары с фотографией", "цена меньше 100 рублей")
        previous_sql: Предыдущий SQL запрос (для retry, опционально)
        error_message: Сообщение об ошибке (для retry, опционально)
        attempt_number: Номер попытки (для retry, опционально)
    
    Returns:
        SQL WHERE условия (без ключевого слова WHERE), готовые для использования в запросе
    """
    db_prompt = await get_prompt("Получить товары при инициализации диалога")
    
    if not db_prompt:
        raise ValueError("Промпт 'Получить товары при инициализации диалога' не найден в БД")
    
    schema_info = """
    СХЕМА БАЗЫ ДАННЫХ: myaso
    
    Таблица: myaso.products
    
    ВАЖНО: Таблица находится в схеме myaso. В SQL запросе используется полное имя myaso.products, но в WHERE условиях указывай только имя колонки БЕЗ схемы и БЕЗ алиаса таблицы.
    
    Поля таблицы products:
    - id (SERIAL) - уникальный идентификатор
    - title (TEXT) - название товара
    - supplier_name (TEXT) - название поставщика
    - from_region (TEXT) - регион происхождения
    - photo (TEXT) - URL фотографии товара (может быть NULL)
    - order_price_kg (NUMERIC) - цена за кг
    - min_order_weight_kg (NUMERIC) - минимальный заказ в кг
    - cooled_or_frozen (TEXT) - "Охлаждённый" или "Замороженный"
    - ready_made (BOOLEAN) - полуфабрикат (true/false)
    - package_type (TEXT) - тип упаковки
    - discount (NUMERIC) - скидка
    
    Примеры правильных WHERE условий:
    - photo IS NOT NULL
    - title ILIKE '%говядина%'
    - order_price_kg < 100
    - cooled_or_frozen = 'Охлаждённый'
    
    НЕ используй алиасы таблиц (p., prod., products.) в условиях!
    НЕ указывай схему (myaso.) в условиях!
    """
    
    system_prompt = db_prompt.format(schema_info=schema_info) if "{schema_info}" in db_prompt else db_prompt
    
    if previous_sql and error_message:
        human_message = f"""Предыдущий SQL запрос (попытка {attempt_number - 1}):
{previous_sql}

Ошибка выполнения:
{error_message}

Попытка {attempt_number}. Исправь SQL запрос и верни только исправленные условия (без WHERE):
{text_conditions}"""
    else:
        human_message = text_conditions
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text_conditions}")
    ])
    
    langchain_settings.setup_langsmith_tracing()
    text2sql_llm = ChatOpenAI(
        model=settings.openrouter.model_id,
        openai_api_key=settings.openrouter.openrouter_api_key,
        openai_api_base=settings.openrouter.base_url,
        temperature=0.1,
    )
    
    chain = prompt | text2sql_llm
    result = await chain.ainvoke({"text_conditions": human_message})
    
    sql_conditions = result.content.strip()
    
    if sql_conditions.startswith("```"):
        lines = sql_conditions.split("\n")
        sql_conditions = "\n".join([line for line in lines if not line.strip().startswith("```")])
        sql_conditions = sql_conditions.strip()
    
    logger.info(f"[generate_sql_from_text] LLM вернул: {sql_conditions[:300]}")
    
    if not sql_conditions or not sql_conditions.strip():
        raise ValueError("LLM не вернул SQL условия")
    
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]
    sql_upper = sql_conditions.upper()
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")
    
    return sql_conditions


@tool
async def execute_sql_conditions(sql_conditions: str, limit: int = 15) -> str:
    """Выполняет SQL запрос с WHERE условиями и возвращает товары.
    
    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE)
        limit: Максимальное количество товаров для возврата
    
    Returns:
        Строка с отформатированным списком найденных товаров
    """
    sql_upper = sql_conditions.upper()
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")
    
    db_dsn = os.getenv("POSTGRES_DSN") or None
    if not db_dsn:
        return "Не настроено подключение к базе данных."
    
    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        
        result = await conn.fetch(
            f"""
            SELECT 
                id,
                title,
                supplier_name,
                from_region,
                photo,
                order_price_kg,
                min_order_weight_kg,
                cooled_or_frozen,
                ready_made,
                package_type,
                discount
            FROM myaso.products
            WHERE {sql_conditions}
            LIMIT $1
            """,
            limit
        )
        json_result = records_to_json(result)
        
        if not json_result:
            return "Товары по указанным условиям не найдены."
        
        products_list = []
        for product in json_result:
            product_info = [
                f"Название: {product.get('title', 'Не указано')}",
                f"Поставщик: {product.get('supplier_name', 'Не указано')}",
                f"Регион: {product.get('from_region', 'Не указано')}",
                f"Цена за кг: {product.get('order_price_kg', 'Не указано')}",
                f"Минимальный заказ (кг): {product.get('min_order_weight_kg', 'Не указано')}",
                f"Охлаждённый/Замороженный: {product.get('cooled_or_frozen', 'Не указано')}",
                f"Полуфабрикат: {'Да' if product.get('ready_made') else 'Нет'}",
                f"Тип упаковки: {product.get('package_type', 'Не указано')}",
                f"Скидка: {product.get('discount', 'Не указано')}",
            ]
            products_list.append("\n".join([info for info in product_info if "Не указано" not in info]))
        
        result_text = "\n\n---\n\n".join(products_list)
        return f"Найдено товаров: {len(json_result)}\n\n{result_text}"
        
    except Exception as e:
        logger.error(f"Ошибка при получении товаров по SQL условиям: {e}")
        logger.error(f"SQL условия, которые вызвали ошибку: {sql_conditions[:200]}")
        raise
    finally:
        if conn:
            await conn.close()


@tool
async def get_product_titles_from_sql_conditions(sql_conditions: str, limit: int = 15) -> List[str]:
    """Получает список названий товаров по SQL WHERE условиям.
    
    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE)
        limit: Максимальное количество товаров
    
    Returns:
        Список названий товаров
    """
    sql_upper = sql_conditions.upper()
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "EXEC", "EXECUTE"]
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")
    
    db_dsn = os.getenv("POSTGRES_DSN") or None
    if not db_dsn:
        return []
    
    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        result = await conn.fetch(
            f"""
            SELECT title
            FROM myaso.products
            WHERE {sql_conditions}
            LIMIT $1
            """,
            limit
        )
        titles = [row["title"] for row in result if row.get("title")]
        return titles
    except Exception as e:
        logger.error(f"Ошибка при получении названий товаров: {e}")
        return []
    finally:
        if conn:
            await conn.close()


@tool
async def get_random_product_titles(limit: int = 10) -> List[str]:
    """Получает список названий случайных товаров.
    
    Args:
        limit: Количество товаров
    
    Returns:
        Список названий товаров
    """
    db_dsn = os.getenv("POSTGRES_DSN") or None
    if not db_dsn:
        return []
    
    conn = None
    try:
        conn = await asyncpg.connect(dsn=db_dsn)
        result = await conn.fetch(
            """
            SELECT title
            FROM myaso.products
            ORDER BY RANDOM()
            LIMIT $1
            """,
            limit
        )
        titles = [row["title"] for row in result if row.get("title")]
        return titles
    except Exception as e:
        logger.error(f"Ошибка при получении случайных названий товаров: {e}")
        return []
    finally:
        if conn:
            await conn.close()


@tool
async def text_to_sql_products(text_conditions: str, limit: int = 15) -> str:
    """Ищет товары по текстовому описанию условий используя text-to-SQL.
    
    Преобразует текстовое описание условий в SQL запрос и возвращает найденные товары.
    
    Используй этот инструмент когда:
    - Пользователь запрашивает товары с конкретными условиями (цена, тип, регион и т.д.)
    - Нужно найти товары по сложным критериям, которые требуют SQL запроса
    - Пользователь описывает условия на русском языке
    
    Args:
        text_conditions: Текстовое описание условий на русском языке (например, "товары с фотографией", "цена меньше 100 рублей")
        limit: Максимальное количество товаров для возврата
    
    Returns:
        Строка с отформатированным списком найденных товаров
    """
    try:
        sql_conditions = await generate_sql_from_text.ainvoke({"text_conditions": text_conditions})
        products_text = await execute_sql_conditions.ainvoke({"sql_conditions": sql_conditions, "limit": limit})
        return products_text
    except Exception as e:
        logger.error(f"Ошибка в text_to_sql_products: {e}")
        return f"Ошибка при поиске товаров: {str(e)}"
