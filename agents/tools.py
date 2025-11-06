"""LangChain Tools.

Содержит функции с декоратором @tool для использования в LangChain агентах.
"""

from __future__ import annotations

from typing import List, Any, Optional
import logging
import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from supabase import (
    create_client,
    ClientOptions,
)

from src.config.settings import settings
from src.config.langchain_settings import LangChainSettings
from src.config.database import get_pool
from src.config.constants import (
    VECTOR_SEARCH_LIMIT,
    DEFAULT_SQL_LIMIT,
    TEXT_TO_SQL_TEMPERATURE,
    DANGEROUS_SQL_KEYWORDS,
    HTTP_TIMEOUT_SECONDS,
    MAX_SQL_RETRY_ATTEMPTS,
)
from src.utils.langchain_retrievers import SupabaseVectorRetriever
from src.utils import records_to_json
from src.utils.prompts import get_prompt
from src.utils.sql_validator import validate_sql_conditions
from src.utils.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
langchain_settings = LangChainSettings()

supabase_client = create_client(
    settings.supabase.supabase_url,
    settings.supabase.supabase_service_key,
    options=ClientOptions(schema="myaso"),
)


@tool
async def vector_search(query: str) -> str:
    """Ищет товары в базе данных по семантическому запросу пользователя.

    Использует векторный поиск (embeddings) для нахождения релевантных товаров.
    Возвращает отформатированный список товаров с их характеристиками.

    ИСПОЛЬЗУЙ ЭТОТ ИНСТРУМЕНТ ДЛЯ БОЛЬШИНСТВА ЗАПРОСОВ О ТОВАРАХ!

    Используй этот инструмент когда:
    - Пользователь запрашивает товары/ассортимент
    - Запрос содержит критерии поиска (тип мяса, часть туши, формат, вес, упаковка)
    - Пользователь спрашивает "Что у вас есть?", "Покажи мясо", "Какие стейки есть?"
    - Пользователь просит рекомендацию с критериями ("Что подходит для гриля?")
    - Пользователь спрашивает про товары от конкретного поставщика ("Что есть из продукции Мироторг", "товары от поставщика X", "что есть от Y")
    - Пользователь спрашивает про товары из конкретного региона ("мясо из региона Z", "товары из Сибири")
    - Запрос содержит название поставщика, региона или другие текстовые атрибуты товара
    - Запрос НЕ содержит числовых условий (цена, вес, скидка с числами)

    НЕ используй если:
    - Запрос содержит только подтверждение/отказ ("Да", "Нет", "Ок")
    - Обсуждаются сервисные темы (доставка, оплата, расписание)
    - Пользователь уточняет детали уже известного товара
    - Запрос содержит ЧИСЛОВЫЕ условия (цена меньше X, вес больше Y, скидка больше Z) - для этого используй generate_sql_from_text + execute_sql_request

    Args:
        query: Текстовый запрос пользователя о товарах/ассортименте

    Returns:
        Строка с отформатированным списком найденных товаров и их характеристиками
    """
    retriever = SupabaseVectorRetriever()

    try:
        documents = await retriever.get_relevant_documents(query, k=VECTOR_SEARCH_LIMIT)
    except Exception as e:
        logger.error(f"Ошибка при поиске по запросу '{query}': {e}", exc_info=True)
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
        products_list.append(
            "\n".join([info for info in product_info if "Не указано" not in info])
        )

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

    supabase = await get_supabase_client()

    for title in product_titles:
        try:
            result = (
                await supabase.table("products")
                .select("*")
                .eq("title", title)
                .execute()
            )

            if not result.data or len(result.data) == 0:
                not_found.append(title)
                continue

            found_with_photo = False
            for product in result.data:
                photo_url = product.get("photo")

                if photo_url:
                    try:
                        async with httpx.AsyncClient(
                            timeout=HTTP_TIMEOUT_SECONDS
                        ) as client:
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
                    except Exception as e:
                        logger.warning(
                            f"[show_product_photos] Ошибка отправки фото для {title}: {e}"
                        )

            if found_with_photo:
                has_photo.append(title)
            else:
                no_photo.append(title)
        except Exception as e:
            logger.error(
                f"[show_product_photos] Ошибка при поиске товара '{title}': {e}"
            )
            not_found.append(title)

    result_parts = []
    if has_photo:
        result_parts.append(
            f"Фотографии следующих товаров отправлены: {', '.join(has_photo)}"
        )
    if no_photo:
        result_parts.append(f"Нет фотографий следующих товаров: {', '.join(no_photo)}")
    if not_found:
        result_parts.append(f"Товары не найдены: {', '.join(not_found)}")

    return (
        "\n".join(result_parts)
        if result_parts
        else "Нет товаров для отправки фотографий."
    )


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
    supabase = await get_supabase_client()

    result = await supabase.table("clients").select("*").eq("phone", phone).execute()
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

    result_text = (
        "\n".join(profile_parts)
        if profile_parts
        else "Профиль найден, но данные отсутствуют."
    )
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
    supabase = await get_supabase_client()

    result = (
        await supabase.table("orders")
        .select("*")
        .eq("client_phone", phone)
        .order("created_at", desc=True)
        .execute()
    )

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
        orders_list.append(
            "\n".join([info for info in order_info if "Не указано" not in info])
        )

    result_text = "\n\n---\n\n".join(orders_list)
    return f"Найдено заказов: {len(result.data)}\n\n{result_text}"


@tool
async def get_random_products(limit: int = 10) -> str:
    """Получает случайные товары из ассортимента.

    FALLBACK инструмент - используй когда другие поиски не дали результатов!

    Используй этот инструмент когда:
    - vector_search вернул "Товары по вашему запросу не найдены"
    - execute_sql_request вернул "Товары по указанным условиям не найдены"
    - Пользователь спрашивает "Что у вас есть?" и нужно показать любой ассортимент
    - Нужно показать примеры товаров из ассортимента
    - Все остальные инструменты поиска не дали результатов

    НЕ используй если:
    - vector_search или execute_sql_request уже нашли товары
    - Есть конкретный запрос, который можно обработать другими инструментами

    Это инструмент последней надежды - используй его только когда ничего не найдено!

    Args:
        limit: Количество товаров для возврата (по умолчанию 10, максимум 20)

    Returns:
        Строка с отформатированным списком случайных товаров
    """
    if limit > 20:
        limit = 20

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
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
                limit,
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
                products_list.append(
                    "\n".join([info for info in product_info if "Не указано" not in info])
                )

            result_text = "\n\n---\n\n".join(products_list)
            return f"Найдено товаров: {len(json_result)}\n\n{result_text}"

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных."
    except Exception as e:
        logger.error(f"Ошибка при получении случайных товаров: {e}")
        return f"Ошибка при получении товаров: {str(e)}"


@tool
async def generate_sql_from_text(
    text_conditions: str,
    topic: Optional[str] = None,
) -> str:
    """Генерирует SQL WHERE условия из текстового описания используя LLM.

    Преобразует текстовое описание условий на русском языке в SQL WHERE условия.
    Имеет встроенный retry механизм (3 попытки с exponential backoff).

    КРИТИЧЕСКИ ВАЖНО: Используй этот инструмент ПЕРВЫМ при наличии ЧИСЛОВЫХ условий в запросе!

    ОБЯЗАТЕЛЬНО используй этот инструмент когда:
    - Запрос содержит ЧИСЛОВЫЕ условия про ЦЕНУ ("цена меньше 80", "дешевле 100 рублей", "цена от 50 до 200", "стоимость меньше X")
    - Запрос содержит ЧИСЛОВЫЕ условия про ВЕС ("вес больше 5 кг", "минимальный заказ меньше 10", "вес от X до Y")
    - Запрос содержит ЧИСЛОВЫЕ условия про СКИДКУ ("скидка больше 15%", "скидка от 10 до 20", "скидка меньше X%")
    - Запрос содержит КОМБИНАЦИЮ числовых условий ("цена меньше 100 и скидка больше 10%", "цена от 50 до 200 и вес меньше 10")
    - Пользователь описывает условия с числами на русском языке ("дешевле 200 рублей", "скидка больше 15%", "цена меньше 80")

    НЕ используй если:
    - Запрос содержит ТОЛЬКО название поставщика БЕЗ чисел ("товары от Мироторг", "продукция X") - используй vector_search
    - Запрос содержит ТОЛЬКО название региона БЕЗ чисел ("мясо из Сибири", "товары из региона Y") - используй vector_search
    - Запрос содержит ТОЛЬКО текстовые критерии БЕЗ чисел ("говядина", "стейки", "полуфабрикаты") - используй vector_search

    После использования этого инструмента, ОБЯЗАТЕЛЬНО используй execute_sql_request для выполнения запроса!

    Args:
        text_conditions: Текстовое описание условий на русском языке (например, "цена меньше 100 рублей", "товары с фотографией и цена от 50 до 200")
        topic: Тема диалога для загрузки промпта из БД (опционально)

    Returns:
        SQL WHERE условия (без ключевого слова WHERE), готовые для использования в execute_sql_request
    """
    db_prompt = None
    if topic:
        db_prompt = await get_prompt(topic)
        if db_prompt:
            logger.info(f"ПРОМПТ: Загружен промпт по topic '{topic}' для generate_sql_from_text")

    if not db_prompt:
        db_prompt = await get_prompt("Получить товары при инициализации диалога")

    if not db_prompt:
        raise ValueError(
            "Промпт 'Получить товары при инициализации диалога' не найден в БД"
        )

    schema_info = """
    СХЕМА БАЗЫ ДАННЫХ: myaso

    Таблица: myaso.products

    ВАЖНО: Таблица находится в схеме myaso. В SQL запросе используется полное имя myaso.products, но в WHERE условиях указывай только имя колонки БЕЗ схемы и БЕЗ алиаса таблицы.

    ПОЛНАЯ СТРУКТУРА ТАБЛИЦЫ products:
    - id (SERIAL/INTEGER) - уникальный идентификатор товара
    - title (TEXT) - название товара
    - supplier_name (TEXT) - название поставщика
    - from_region (TEXT) - регион происхождения товара
    - photo (TEXT) - URL фотографии товара (может быть NULL)
    - pricelist_date (DATE/TIMESTAMP) - дата прайс-листа
    - package_weight (NUMERIC) - вес упаковки в кг
    - order_price_kg (NUMERIC) - цена за кг
    - min_order_weight_kg (NUMERIC) - минимальный заказ в кг
    - discount (NUMERIC) - скидка в процентах или абсолютном значении
    - ready_made (BOOLEAN) - полуфабрикат (true/false, NULL возможен)
    - package_type (TEXT) - тип упаковки
    - cooled_or_frozen (TEXT) - состояние товара: "Охлаждённый" или "Замороженный" (может быть NULL)
    - product_in_package (TEXT или NUMERIC) - количество товара в упаковке

    ПРАВИЛА ДЛЯ WHERE УСЛОВИЙ:
    1. Используй только имена колонок БЕЗ префиксов (title, а не products.title или myaso.products.title)
    2. Для текстовых полей используй ILIKE для поиска без учета регистра: title ILIKE '%говядина%'
    3. Для проверки NULL используй: photo IS NOT NULL или photo IS NULL
    4. Для булевых полей: ready_made = true или ready_made = false
    5. Для числовых сравнений: order_price_kg < 100, discount >= 15
    6. Для точного совпадения текста: supplier_name = 'Мироторг' или cooled_or_frozen = 'Охлаждённый'
    7. Для дат: pricelist_date > '2024-01-01' или pricelist_date >= CURRENT_DATE

    НЕ используй алиасы таблиц (p., prod., products.) в условиях!
    НЕ указывай схему (myaso.) в условиях!
    НЕ используй ключевое слово WHERE в ответе - только условия!
    """
    system_prompt = f"{db_prompt}\n\n{schema_info}"


    max_attempts = MAX_SQL_RETRY_ATTEMPTS
    previous_sql = None
    last_error = None

    langchain_settings.setup_langsmith_tracing()
    text2sql_llm = ChatOpenAI(
        model=settings.openrouter.model_id,
        openai_api_key=settings.openrouter.openrouter_api_key,
        openai_api_base=settings.openrouter.base_url,
        temperature=TEXT_TO_SQL_TEMPERATURE,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1 and previous_sql and last_error:
                human_message = f"""Предыдущий SQL запрос (попытка {attempt - 1}):
{previous_sql}

Ошибка выполнения:
{last_error}

Попытка {attempt}. Исправь SQL запрос и верни только исправленные условия (без WHERE):
{text_conditions}"""
            else:
                human_message = text_conditions

            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{text_conditions}")]
            )
            chain = prompt | text2sql_llm
            result = await chain.ainvoke({"text_conditions": human_message})

            sql_conditions = result.content.strip()

            if sql_conditions.startswith("```"):
                lines = sql_conditions.split("\n")
                sql_conditions = "\n".join(
                    [line for line in lines if not line.strip().startswith("```")]
                )
                sql_conditions = sql_conditions.strip()

            if not sql_conditions or not sql_conditions.strip():
                raise ValueError("LLM вернул пустые SQL условия")

            sql_upper = sql_conditions.upper()
            for keyword in DANGEROUS_SQL_KEYWORDS:
                if keyword in sql_upper:
                    logger.error(
                        f"Обнаружена опасная SQL команда: {keyword} в запросе: {sql_conditions[:200]}"
                    )
                    raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")

            return sql_conditions

        except Exception as e:
            last_error = str(e)
            previous_sql = sql_conditions if 'sql_conditions' in locals() else None

            logger.warning(
                f"[generate_sql_from_text] Попытка {attempt}/{max_attempts} не удалась: {e}",
                exc_info=attempt == max_attempts,
            )

            if attempt < max_attempts:
                import asyncio
                wait_time = 2 ** (attempt - 1)
                await asyncio.sleep(wait_time)
            else:
                error_msg = (
                    f"Не удалось сгенерировать SQL условия после {max_attempts} попыток. "
                    f"Последняя ошибка: {last_error}"
                )
                logger.error(f"[generate_sql_from_text] {error_msg}")
                raise ValueError(error_msg) from e

    raise ValueError("Не удалось сгенерировать SQL условия")


@tool
async def execute_sql_request(
    sql_conditions: str, limit: int = DEFAULT_SQL_LIMIT
) -> str:
    """Выполняет SQL запрос с WHERE условиями и возвращает товары.

    Используй этот инструмент когда:
    - У тебя есть готовые SQL WHERE условия (сгенерированные через generate_sql_from_text)
    - Нужно выполнить SQL запрос для поиска товаров по условиям
    - После успешной генерации SQL условий нужно получить результаты

    НЕ используй если:
    - У тебя нет готовых SQL условий - сначала используй generate_sql_from_text
    - Запрос не содержит числовых условий - используй vector_search

    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE)
        limit: Максимальное количество товаров для возврата

    Returns:
        Строка с отформатированным списком найденных товаров
    """
    sql_conditions = sql_conditions.strip()

    try:
        validate_sql_conditions(sql_conditions)
    except ValueError as e:
        logger.error(f"SQL условия не прошли валидацию: {e}. Условия: {sql_conditions[:200]}")
        raise

    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = f"""
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
            """
            result = await conn.fetch(query, limit)
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
                products_list.append(
                    "\n".join([info for info in product_info if "Не указано" not in info])
                )

            result_text = "\n\n---\n\n".join(products_list)
            return f"Найдено товаров: {len(json_result)}\n\n{result_text}"

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных."
    except Exception as e:
        logger.error(f"Ошибка при получении товаров по SQL условиям: {e}", exc_info=True)
        logger.error(f"SQL условия, которые вызвали ошибку: {sql_conditions[:200]}")
        return "Товары по указанным условиям не найдены."


