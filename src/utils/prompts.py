"""Утилиты для работы с промптами из базы данных."""

from __future__ import annotations

from typing import Optional, Dict, Any
import logging
import re

from src.utils import get_supabase_client

logger = logging.getLogger(__name__)


async def get_prompt(topic: str) -> Optional[str]:
    """Получает промпт из таблицы myaso.prompts по topic.

    Args:
        topic: Значение колонки topic из таблицы prompts (например, "Продать", "Узнать потребность")

    Returns:
        Текст промпта из колонки prompt или None, если промпт не найден
    """
    try:
        supabase = await get_supabase_client()

        result = (
            await supabase.table("prompts")
            .select("prompt")
            .eq("topic", topic)
            .execute()
        )

        if result.data and len(result.data) > 0:
            row = result.data[0]
            return row.get("prompt")

        return None
    except Exception as e:
        logger.error(f"Ошибка при получении промпта для topic '{topic}': {e}")
        return None


async def get_system_value(topic: str) -> Optional[str]:
    """Получает значение из таблицы myaso.system по topic.

    Args:
        topic: Название параметра системы (например, "Наценка на кг/руб (>100 руб)")

    Returns:
        Значение параметра или None, если параметр не найден
    """
    try:
        supabase = await get_supabase_client()

        result = (
            await supabase.table("system").select("value").eq("topic", topic).execute()
        )

        if result.data and len(result.data) > 0:
            return result.data[0].get("value")

        return None
    except Exception as e:
        logger.error(f"Ошибка при получении значения системы '{topic}': {e}")
        return None


async def get_all_system_values() -> Dict[str, str]:
    """Получает ВСЕ значения из таблицы myaso.system.

    Всегда возвращает словарь (не None), даже если записей нет или произошла ошибка.
    В случае ошибки возвращает пустой словарь.

    Returns:
        Словарь, где ключ - это topic, значение - это value.
        Если записей нет или произошла ошибка, возвращает пустой словарь {}.
    """
    try:
        supabase = await get_supabase_client()

        result = await supabase.table("system").select("topic, value").execute()

        if result.data:
            return {row.get("topic", ""): row.get("value", "") for row in result.data}

        return {}
    except Exception as e:
        logger.error(f"Ошибка при получении всех значений системы: {e}")
        return {}


def format_system_variables(system_vars: Dict[str, str]) -> str:
    """Форматирует системные переменные в строку для промпта.

    Args:
        system_vars: Словарь системных переменных (topic -> value)

    Returns:
        Отформатированная строка с системными переменными
    """
    if not system_vars:
        return "No system variables available"

    lines = []
    for topic, value in system_vars.items():
        lines.append(f"{topic}: {value}")

    return "\n".join(lines)


def escape_prompt_variables(prompt: str) -> str:
    """Экранирует переменные в промпте, которые не являются шаблонными переменными LangChain.

    LangChain ChatPromptTemplate ожидает только определенные переменные:
    - input
    - chat_history
    - agent_scratchpad
    - intermediate_steps

    Все остальные фигурные скобки должны быть экранированы двойными скобками,
    чтобы они трактовались как буквальный текст, а не как переменные шаблона.

    Args:
        prompt: Текст промпта, который может содержать переменные в фигурных скобках

    Returns:
        Промпт с экранированными переменными (кроме известных шаблонных переменных)
    """
    known_variables = {
        "input",
        "chat_history",
        "agent_scratchpad",
        "intermediate_steps",
    }

    pattern = r"(?<!\{)\{([^}]+)\}(?!\})"

    def replace_var(match):
        var_name = match.group(1).strip()
        if var_name in known_variables:
            return match.group(0)
        return f"{{{{{var_name}}}}}"

    escaped_prompt = re.sub(pattern, replace_var, prompt)

    return escaped_prompt


def build_prompt_with_context(
    base_prompt: str,
    client_info: Optional[str] = None,
    system_vars: Optional[Dict[str, str]] = None,
) -> str:
    """Строит промпт с контекстом клиента и системными переменными.

    Формат промпта:
    ==========================================================================================================
    CLIENT INFO: {client_info} (только если client_info is not None)
    ==========================================================================================================
    SYS VARIABLES: {system_vars или "No system variables available"} (всегда показывается)
    ==========================================================================================================

    {base_prompt}

    Args:
        base_prompt: Базовый промпт из БД
        client_info: Информация о клиенте (опционально, если None - блок не показывается)
        system_vars: Словарь системных переменных (опционально, если None - показывается "No system variables available")

    Returns:
        Полный промпт с контекстом (с экранированными переменными)
    """
    separator = "=" * 100

    parts = []

    if client_info is not None:
        parts.append(f"{separator}\n")
        parts.append(f"CLIENT INFO: {client_info}\n")
        parts.append(f"{separator}\n")

    parts.append(f"{separator}\n")
    if system_vars is not None and system_vars:
        system_vars_text = format_system_variables(system_vars)
        parts.append(f"SYSTEM VARIABLES: {system_vars_text}\n")
    else:
        parts.append(f"SYSTEM VARIABLES: No system variables available\n")
    parts.append(f"{separator}\n")

    if parts:
        parts.append("\n")

    parts.append(base_prompt)

    full_prompt = "".join(parts)
    
    escaped_prompt = escape_prompt_variables(full_prompt)

    return escaped_prompt
