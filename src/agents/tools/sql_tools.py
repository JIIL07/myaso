"""Инструменты для работы с SQL запросами."""

from __future__ import annotations

from typing import Optional
import logging
import asyncio
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import settings
from src.config.langchain_settings import LangChainSettings
from src.config.constants import (
    DEFAULT_SQL_LIMIT,
    TEXT_TO_SQL_TEMPERATURE,
    DANGEROUS_SQL_KEYWORDS,
    MAX_SQL_RETRY_ATTEMPTS,
)
from src.utils.prompts import get_prompt
from src.utils import validate_sql_conditions
from src.database.queries.products_queries import get_products_by_sql_conditions
from src.utils import records_to_json

logger = logging.getLogger(__name__)
langchain_settings = LangChainSettings()


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
        json_result = await get_products_by_sql_conditions(sql_conditions, limit)

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

