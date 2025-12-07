"""Утилиты для работы с правилами из базы данных."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from src.config.database_constants import (
    COLUMN_RULE_NAME,
    COLUMN_RULE_TYPE,
    COLUMN_RULE_VALUE,
    TABLE_RULES,
)
from src.utils.supabase_client import get_supabase_client
from src.utils.exceptions import RuleNotFoundError

logger = logging.getLogger(__name__)

_RULE_CACHE: Dict[str, tuple[Dict[str, Any], float]] = {}
_RULE_CACHE_TTL = 600


async def get_rule(rule_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """Получает правило из таблицы myaso.rules по имени с кэшированием.

    Args:
        rule_name: Имя правила (например, "MAX_AGENT_ITERATIONS", "DEFAULT_TEMPERATURE")
        use_cache: Использовать ли кэш (по умолчанию True)

    Returns:
        Словарь с данными правила (rule_name, rule_type, rule_value, description, category)
        или None, если правило не найдено

    Raises:
        RuleNotFoundError: Если правило не найдено в БД
    """
    current_time = time.time()

    if use_cache and rule_name in _RULE_CACHE:
        cached_rule, cache_time = _RULE_CACHE[rule_name]
        if current_time - cache_time < _RULE_CACHE_TTL:
            return cached_rule.copy()
        del _RULE_CACHE[rule_name]

    try:
        supabase = await get_supabase_client()

        result = (
            await supabase.table(TABLE_RULES)
            .select("*")
            .eq(COLUMN_RULE_NAME, rule_name)
            .execute()
        )

        if result.data and len(result.data) > 0:
            rule_data = result.data[0]
            _RULE_CACHE[rule_name] = (rule_data.copy(), current_time)
            return rule_data

        raise RuleNotFoundError(
            f"Правило '{rule_name}' не найдено в базе данных",
            {"rule_name": rule_name}
        )
    except RuleNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении правила '{rule_name}': {e}")
        raise RuleNotFoundError(
            f"Ошибка при загрузке правила '{rule_name}': {e}",
            {"rule_name": rule_name, "error": str(e)}
        ) from e


async def get_rule_value(rule_name: str) -> Any:
    """Получает значение правила с автоматической типизацией.

    Args:
        rule_name: Имя правила

    Returns:
        Значение правила с правильным типом (int, float, bool, list, str)

    Raises:
        RuleNotFoundError: Если правило не найдено
    """
    rule = await get_rule(rule_name)
    if not rule:
        raise RuleNotFoundError(
            f"Правило '{rule_name}' не найдено",
            {"rule_name": rule_name}
        )

    rule_type = rule.get(COLUMN_RULE_TYPE)
    rule_value = rule.get(COLUMN_RULE_VALUE)

    if rule_type == "list":
        try:
            return json.loads(rule_value)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Ошибка парсинга JSON для правила '{rule_name}': {e}")
            raise ValueError(f"Неверный формат JSON для правила '{rule_name}': {rule_value}") from e
    elif rule_type == "boolean":
        value_lower = str(rule_value).lower().strip()
        return value_lower in ("true", "1", "yes", "on")
    elif rule_type in ("limit", "timeout"):
        try:
            if "." in str(rule_value):
                return float(rule_value)
            return int(rule_value)
        except (ValueError, TypeError) as e:
            logger.error(f"Ошибка преобразования числа для правила '{rule_name}': {e}")
            raise ValueError(f"Неверный формат числа для правила '{rule_name}': {rule_value}") from e
    else:
        return str(rule_value)


async def get_rule_as_int(rule_name: str) -> int:
    """Получает правило как целое число.

    Args:
        rule_name: Имя правила

    Returns:
        Значение правила как int

    Raises:
        RuleNotFoundError: Если правило не найдено
        ValueError: Если значение не может быть преобразовано в int
    """
    value = await get_rule_value(rule_name)
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Правило '{rule_name}' не может быть преобразовано в int: {value}") from e


async def get_rule_as_float(rule_name: str) -> float:
    """Получает правило как число с плавающей точкой.

    Args:
        rule_name: Имя правила

    Returns:
        Значение правила как float

    Raises:
        RuleNotFoundError: Если правило не найдено
        ValueError: Если значение не может быть преобразовано в float
    """
    value = await get_rule_value(rule_name)
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Правило '{rule_name}' не может быть преобразовано в float: {value}") from e


async def get_rule_as_bool(rule_name: str) -> bool:
    """Получает правило как булево значение.

    Args:
        rule_name: Имя правила

    Returns:
        Значение правила как bool

    Raises:
        RuleNotFoundError: Если правило не найдено
    """
    value = await get_rule_value(rule_name)
    if isinstance(value, bool):
        return value
    value_lower = str(value).lower().strip()
    return value_lower in ("true", "1", "yes", "on")


async def get_rule_as_list(rule_name: str) -> List[str]:
    """Получает правило как список строк.

    Args:
        rule_name: Имя правила

    Returns:
        Значение правила как List[str] (парсится из JSON)

    Raises:
        RuleNotFoundError: Если правило не найдено
        ValueError: Если значение не может быть распарсено как JSON список
    """
    value = await get_rule_value(rule_name)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError(f"Правило '{rule_name}' не является списком: {value}")


async def get_rule_as_str(rule_name: str) -> str:
    """Получает правило как строку.

    Args:
        rule_name: Имя правила

    Returns:
        Значение правила как str

    Raises:
        RuleNotFoundError: Если правило не найдено
    """
    value = await get_rule_value(rule_name)
    return str(value)


async def get_all_instruction_rules() -> Dict[str, str]:
    """Получает все инструкции (правила с типом 'instruction') из БД.

    Returns:
        Словарь {rule_name: rule_value} со всеми инструкциями
    """
    try:
        supabase = await get_supabase_client()

        result = (
            await supabase.table(TABLE_RULES)
            .select("*")
            .eq(COLUMN_RULE_TYPE, "instruction")
            .execute()
        )

        if not result.data:
            return {}

        instructions = {}
        current_time = time.time()

        for row in result.data:
            rule_name = row.get(COLUMN_RULE_NAME)
            rule_value = row.get(COLUMN_RULE_VALUE)

            if rule_name and rule_value:
                _RULE_CACHE[rule_name] = (row.copy(), current_time)
                instructions[rule_name] = str(rule_value)

        return instructions
    except Exception as e:
        logger.error(f"Ошибка при получении всех инструкций: {e}")
        return {}
