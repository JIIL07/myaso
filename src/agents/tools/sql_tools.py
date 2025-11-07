"""Инструменты для работы с SQL запросами."""

from __future__ import annotations

from typing import Optional
import json
import logging
import asyncio
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config.settings import settings
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


def get_products_table_schema() -> str:
    """Get exact products table schema for SQL generation.

    Returns:
        Строка с точной схемой таблицы products для использования в SQL генерации
    """
    return """
TABLE: products

COLUMNS:
- id (int8) - primary key
- title (text) - название товара "Грудинка Премиум"
- from_region (text) - регион "Бурятия", "Сибирь"
- photo (text) - URL фото
- pricelist_date (date) - дата прайслиста
- supplier_name (text) - поставщик
- delivery_cost_MSK (float8) - доставка до Москвы
- package_weight (float8) - вес упаковки кг
- prepayment_1t (int8) - предоплата за тонну
- order_price_kg (float8) - ЦЕНА ЗА КГ в рублях
- min_order_weight_kg (int8) - МИНИМАЛЬНЫЙ ЗАКАЗ в кг
- discount (text) - скидка
- ready_made (bool) - готовый продукт
- package_type (text) - тип упаковки
- cooled_or_frozen (text) - охлажденный/замороженный
- product_in_package (text) - продукт в упаковке
- embedding (vector) - НЕ ИСПОЛЬЗУЙ в WHERE!

ПРАВИЛА:
1. Цена: order_price_kg < 100
2. Регион: from_region = 'Бурятия'
3. Минимальный заказ: min_order_weight_kg < 10
4. Тип: cooled_or_frozen = 'охлажденное'
5. Поставщик: supplier_name LIKE '%Мироторг%'
6. С фото: photo IS NOT NULL AND photo != ''

ПРИМЕРЫ:
"дешевле 100" → order_price_kg < 100
"из Бурятии" → from_region = 'Бурятия'
"минимальный заказ меньше 15" → min_order_weight_kg < 15
"""


@tool
async def generate_sql_from_text(
    text_conditions: str,
    topic: Optional[str] = None,
) -> str:
    """Генерирует SQL WHERE условия из текстового описания на русском языке.

    Преобразует текстовые условия в SQL WHERE условия для поиска товаров.
    Использует LLM для понимания запроса и генерации правильного SQL.
    Имеет встроенный retry механизм (3 попытки с exponential backoff).

    ════════════════════════════════════════════════════════════════════════════════
    КОГДА ИСПОЛЬЗОВАТЬ:
    ════════════════════════════════════════════════════════════════════════════════

    ✅ ОБЯЗАТЕЛЬНО используй для:
    - Числовые условия по ЦЕНЕ: "цена меньше 80", "дешевле 100 рублей", "цена от 50 до 200"
    - Числовые условия по ВЕСУ: "вес больше 5 кг", "минимальный заказ меньше 10"
    - Числовые условия по СКИДКЕ: "скидка больше 15%", "скидка от 10 до 20"
    - Комбинации числовых условий: "цена меньше 100 и скидка больше 10%"
    - Пустые запросы или init_conversation → передай описание темы/категории в text_conditions

    ❌ НЕ используй для:
    - Только название поставщика БЕЗ чисел: "товары от Мироторг" → используй vector_search
    - Только название региона БЕЗ чисел: "мясо из Сибири" → используй vector_search
    - Только текстовые критерии БЕЗ чисел: "говядина", "стейки" → используй vector_search

    ════════════════════════════════════════════════════════════════════════════════
    ВАЖНО - ПОСЛЕ ИСПОЛЬЗОВАНИЯ:
    ════════════════════════════════════════════════════════════════════════════════

    После вызова generate_sql_from_text ОБЯЗАТЕЛЬНО вызови execute_sql_request с полученными SQL условиями!

    Алгоритм работы:
    1. generate_sql_from_text("цена меньше 100") → получаешь SQL условия
    2. execute_sql_request(sql_conditions) → получаешь товары

    ════════════════════════════════════════════════════════════════════════════════
    КРИТИЧЕСКИ ВАЖНО - ЗАПРЕТ НА СЛОВА-ДЕЙСТВИЯ:
    ════════════════════════════════════════════════════════════════════════════════

    ⚠️ НИКОГДА не используй слова-действия (продать, купить, показать, найти, есть, 
    отправить и т.д.) в SQL условиях для поиска по title!

    Если запрос содержит ТОЛЬКО слова-действия без конкретных критериев:
    → Верни: photo IS NOT NULL AND photo != ''

    Если запрос содержит слова-действия + конкретные критерии:
    → Игнорируй слова-действия, используй ТОЛЬКО конкретные критерии товаров

    Примеры:
    - "продать" → photo IS NOT NULL AND photo != '' (НЕ title ILIKE '%продать%'!)
    - "продать говядину" → title ILIKE '%говядина%' AND photo IS NOT NULL AND photo != ''
    - "есть коралл" → supplier_name = 'Коралл' AND photo IS NOT NULL AND photo != ''

    Args:
        text_conditions: Текстовое описание условий на русском языке
                        Примеры: "цена меньше 100 рублей", "товары с фото и цена от 50 до 200"

    Returns:
        SQL WHERE условия (без ключевого слова WHERE) для использования в execute_sql_request
    """
    db_prompt = None
    if topic:
        db_prompt = await get_prompt(topic)

    if not db_prompt:
        db_prompt = await get_prompt("Получить товары при инициализации диалога")

    if not db_prompt:
        raise ValueError(
            "Промпт 'Получить товары при инициализации диалога' не найден в БД"
        )

    schema_info = f"""
    СХЕМА БАЗЫ ДАННЫХ: myaso

    {get_products_table_schema()}

    СЦЕНАРИЙ 1: Запрос содержит ТОЛЬКО слова-действия БЕЗ конкретных критериев товаров
    Примеры: "продать", "показать товары", "что у вас есть", "есть", "дай", "покажи"
    
    → Верни ТОЛЬКО условие для фото: photo IS NOT NULL AND photo != ''
    → НЕ добавляй никаких условий по title!
    → НЕ ищи товары по словам-действиям!

    СЦЕНАРИЙ 2: Запрос содержит слова-действия + конкретные критерии товаров
    Примеры: "продать говядину", "показать стейки", "есть коралл", "найти мясо из Бурятии"
    
    → ПОЛНОСТЬЮ ИГНОРИРУЙ слова-действия (продать, показать, есть, найти)
    → Используй ТОЛЬКО конкретные критерии товаров:
      * "продать говядину" → title ILIKE '%говядина%' AND photo IS NOT NULL AND photo != ''
      * "показать стейки" → title ILIKE '%стейки%' AND photo IS NOT NULL AND photo != ''
      * "есть коралл" → supplier_name = 'Коралл' AND photo IS NOT NULL AND photo != ''
      * "найти мясо из Бурятии" → from_region = 'Бурятия' AND photo IS NOT NULL AND photo != ''

    ПРАВИЛА ДЛЯ WHERE УСЛОВИЙ:
    1. Используй ТОЛЬКО колонки из списка выше! Никаких других колонок не существует!
    2. Используй только имена колонок БЕЗ префиксов (title, а не products.title или myaso.products.title)
    3. Для текстовых полей используй ILIKE для поиска без учета регистра: title ILIKE '%говядина%'
    4. Для проверки NULL используй: photo IS NOT NULL или photo IS NULL
    5. Для булевых полей: ready_made = true или ready_made = false
    6. Для числовых сравнений: order_price_kg < 100, discount >= 15
    7. Для точного совпадения текста: supplier_name = 'Мироторг' или cooled_or_frozen = 'Охлаждённый'
    8. Для дат: pricelist_date > '2024-01-01' или pricelist_date >= CURRENT_DATE

    ВАЖНО - ТОВАРЫ С ФОТО:
    ВСЕГДА добавляй условие для выбора только товаров с фотографиями: photo IS NOT NULL AND photo != ''
    Это условие должно быть в КАЖДОМ SQL запросе, если пользователь явно не просит товары без фото.
    Пример: order_price_kg < 100 AND photo IS NOT NULL AND photo != ''
    """
    system_prompt = f"{db_prompt}\n\n{schema_info}"

    max_attempts = MAX_SQL_RETRY_ATTEMPTS
    previous_sql = None
    last_error = None

    text2sql_llm = ChatOpenAI(
        model=settings.openrouter.model_id,
        openai_api_key=settings.openrouter.openrouter_api_key,
        openai_api_base=settings.openrouter.base_url,
        temperature=TEXT_TO_SQL_TEMPERATURE,
    )

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1 and previous_sql and last_error:
                error_hint = ""

                error_lower = last_error.lower()

                if "неразрешенных колонок" in error_lower or "column" in error_lower and "does not exist" in error_lower:
                    error_hint = f"""

ОШИБКА: Использована несуществующая колонка!
Предыдущий SQL (попытка {attempt - 1}): {previous_sql}
Ошибка: {last_error}

ИСПРАВЛЕНИЕ:
1. Проверь каждую колонку в SQL - используй ТОЛЬКО колонки из схемы: id, title, supplier_name, from_region, photo, pricelist_date, package_weight, order_price_kg, min_order_weight_kg, discount, ready_made, package_type, cooled_or_frozen, product_in_package
2. НЕ используй: topic, category, name, description - этих колонок НЕТ!
3. Если нужно найти товары по теме - используй title ILIKE '%тема%'
4. Удали все условия с несуществующими колонками

Точная схема таблицы:
{get_products_table_schema()}
"""
                elif "syntax" in error_lower or "синтаксис" in error_lower:
                    error_hint = f"""

ОШИБКА СИНТАКСИСА SQL!
Предыдущий SQL (попытка {attempt - 1}): {previous_sql}
Ошибка: {last_error}

ИСПРАВЛЕНИЕ:
1. Проверь синтаксис SQL - используй правильные операторы (=, <, >, <=, >=, LIKE, ILIKE, IS NULL, IS NOT NULL)
2. Для текста используй кавычки: supplier_name = 'Мироторг'
3. Для чисел НЕ используй кавычки: order_price_kg < 100
4. НЕ используй ключевое слово WHERE - только условия!
5. Используй AND/OR для объединения условий
"""
                else:
                    error_hint = f"""

ОШИБКА ВЫПОЛНЕНИЯ SQL!
Предыдущий SQL (попытка {attempt - 1}): {previous_sql}
Ошибка: {last_error}

ИСПРАВЛЕНИЕ:
1. Проверь все условия на корректность
2. Убедись что используешь только существующие колонки
3. Проверь типы данных (текст в кавычках, числа без кавычек)
4. Используй правильные операторы сравнения

Схема таблицы:
{get_products_table_schema()}
"""

                human_message = f"""ИСПРАВЬ SQL ЗАПРОС!

Исходный запрос: {text_conditions}
{error_hint}
Попытка {attempt}/{max_attempts}. Верни ТОЛЬКО исправленные SQL условия (без WHERE, без SELECT, только условия для WHERE):
"""
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

            if sql_conditions.upper().startswith("WHERE"):
                sql_conditions = sql_conditions[5:].strip()

            if not sql_conditions or not sql_conditions.strip():
                raise ValueError("LLM вернул пустые SQL условия")

            sql_upper = sql_conditions.upper()
            for keyword in DANGEROUS_SQL_KEYWORDS:
                if keyword in sql_upper:
                    logger.error(
                        f"Обнаружена опасная SQL команда: {keyword} в запросе: {sql_conditions[:200]}"
                    )
                    raise ValueError(f"Обнаружена опасная SQL команда: {keyword}")

            try:
                validate_sql_conditions(sql_conditions)
            except ValueError as validation_error:
                last_error = f"Валидация SQL не прошла: {validation_error}"
                previous_sql = sql_conditions
                if attempt < max_attempts:
                    wait_time = 2 ** (attempt - 1)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise

            return sql_conditions

        except ValueError as ve:
            last_error = str(ve)
            previous_sql = sql_conditions if 'sql_conditions' in locals() else None

            logger.warning(
                f"[generate_sql_from_text] Попытка {attempt}/{max_attempts} не удалась: {ve}",
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
                raise ValueError(error_msg) from ve
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

    Выполняет поиск товаров по SQL WHERE условиям, сгенерированным через generate_sql_from_text.
    Возвращает до 50 товаров в компактном формате с ID для последующей отправки фото.

    ════════════════════════════════════════════════════════════════════════════════
    КОГДА ИСПОЛЬЗОВАТЬ:
    ════════════════════════════════════════════════════════════════════════════════

    ✅ Используй когда:
    - У тебя есть готовые SQL WHERE условия от generate_sql_from_text
    - Нужно выполнить SQL запрос для поиска товаров по числовым условиям
    - После успешной генерации SQL условий нужно получить результаты

    ❌ НЕ используй если:
    - У тебя нет готовых SQL условий → сначала используй generate_sql_from_text
    - Запрос не содержит числовых условий → используй vector_search

    ════════════════════════════════════════════════════════════════════════════════
    ВАЖНО - ПОСЛЕДОВАТЕЛЬНОСТЬ:
    ════════════════════════════════════════════════════════════════════════════════

    Всегда используй в паре с generate_sql_from_text:
    1. generate_sql_from_text("цена меньше 100") → получаешь SQL условия
    2. execute_sql_request(sql_conditions) → получаешь товары

    ════════════════════════════════════════════════════════════════════════════════
    ФОРМАТ ОТВЕТА:
    ════════════════════════════════════════════════════════════════════════════════

    Ответ содержит:
    - Количество найденных товаров
    - Список товаров в структурированном формате (каждый товар на отдельной строке с отступами)
    - Предупреждение если есть ещё товары (показываем первые 50)
    - Секцию [PRODUCT_IDS] с ID товаров для последующей отправки фото

    Args:
        sql_conditions: SQL WHERE условия (без ключевого слова WHERE), полученные от generate_sql_from_text
        limit: Максимальное количество товаров для возврата (по умолчанию 50)

    Returns:
        Строка с отформатированным списком найденных товаров (до 50) и их ID в секции [PRODUCT_IDS]
    """
    sql_conditions = sql_conditions.strip()

    try:
        validate_sql_conditions(sql_conditions)
    except ValueError as e:
        logger.error(f"SQL условия не прошли валидацию: {e}. Условия: {sql_conditions[:200]}")
        raise

    try:
        json_result, has_more = await get_products_by_sql_conditions(sql_conditions, limit)

        if not json_result:
            return "Товары по указанным условиям не найдены."

        products_list = []
        product_ids = []
        for product in json_result:
            product_id = product.get('id')
            if product_id:
                product_ids.append(product_id)

            title = product.get('title', 'Не указано')
            supplier = product.get('supplier_name', '')
            price = product.get('order_price_kg', '')
            region = product.get('from_region', '')
            
            product_lines = [f"📦 {title}"]
            if supplier and supplier != 'Не указано':
                product_lines.append(f"   Поставщик: {supplier}")
            if price and price != 'Не указано':
                product_lines.append(f"   Цена: {price}₽/кг")
            if region and region != 'Не указано':
                product_lines.append(f"   Регион: {region}")
            
            products_list.append("\n".join(product_lines))

        result_text = "\n\n".join(products_list)
        more_text = "\n\n⚠️ В базе данных есть ещё товары, показываем первые 50. Используйте более конкретные критерии поиска для уточнения." if has_more else ""

        ids_json = json.dumps({"product_ids": product_ids}) if product_ids else ""
        ids_section = f"\n\n[PRODUCT_IDS]{ids_json}[/PRODUCT_IDS]" if ids_json else ""

        return f"Найдено товаров: {len(json_result)}{more_text}\n\n{result_text}{ids_section}"

    except RuntimeError as e:
        logger.error(f"Ошибка подключения к базе данных: {e}")
        return "Не настроено подключение к базе данных."
    except Exception as e:
        logger.error(f"Ошибка при получении товаров по SQL условиям: {e}", exc_info=True)
        logger.error(f"SQL условия, которые вызвали ошибку: {sql_conditions[:200]}")
        return "Товары по указанным условиям не найдены."

