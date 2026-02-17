"""Сервис для работы с промптами и контекстом."""
import inspect
import logging
import re
import time
from typing import Any, Dict, List, Optional

from src.services.database.constants import (
    COLUMN_TOPIC,
    COLUMN_VALUE,
    TABLE_SYSTEM,
)
from src.services.database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_PROMPT_CACHE: Dict[str, tuple[str, float]] = {}
_PROMPT_CACHE_TTL = 600

_langfuse_prompt_service: Optional[Any] = None
_langfuse_lock = None  # Инициализируется при первом использовании


def _get_langfuse_service() -> Optional[Any]:
    """Получает экземпляр Langfuse prompt service (ленивая инициализация).
    
    Использует threading.Lock для thread-safety при параллельных запросах.
    """
    global _langfuse_prompt_service, _langfuse_lock
    
    if _langfuse_prompt_service is not None:
        return _langfuse_prompt_service
    
    # Инициализация блокировки при первом использовании
    if _langfuse_lock is None:
        import threading
        _langfuse_lock = threading.Lock()
    
    with _langfuse_lock:
        # Double-check после получения блокировки
        if _langfuse_prompt_service is not None:
            return _langfuse_prompt_service
        
        try:
            from src.config.settings import settings
            from src.services.langfuse.prompt_service import LangfusePromptService
            
            if hasattr(settings, 'langfuse'):
                _langfuse_prompt_service = LangfusePromptService(config=settings.langfuse)
                return _langfuse_prompt_service
        except Exception as e:
            logger.debug(f"Не удалось инициализировать Langfuse prompt service: {e}")
    
    return None


def _get_calling_context() -> Dict[str, Any]:
    """Получает контекст вызова для логирования.
    
    Returns:
        Словарь с информацией о вызывающей функции и модуле
    """
    frame = inspect.currentframe()
    try:
        # Пропускаем текущий фрейм и фрейм get_prompt/compose_prompts
        caller_frame = frame.f_back.f_back if frame.f_back else None
        if caller_frame:
            return {
                "calling_module": caller_frame.f_globals.get("__name__", "unknown"),
                "calling_function": caller_frame.f_code.co_name,
                "calling_file": caller_frame.f_code.co_filename.split("/")[-1],
                "calling_line": caller_frame.f_lineno,
            }
    except Exception:
        pass
    finally:
        del frame
    return {}


async def get_prompt(
    prompt_name: str,
    default_prompt: Optional[str] = None,
    use_cache: bool = True,
    langfuse_label: Optional[str] = "production",
    variables: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> Optional[str]:
    """Получает промпт из Langfuse по названию с кэшированием.

    Args:
        prompt_name: Название промпта в Langfuse (например, "welcome-message", "system-prompt")
        default_prompt: Дефолтный промпт, который будет возвращен, если промпт не найден в Langfuse
        use_cache: Использовать ли кэш (по умолчанию True)
        langfuse_label: Метка для фильтрации версий в Langfuse (по умолчанию "production")
        variables: Переменные для подстановки в промпт из Langfuse
        context: Дополнительный контекст использования промпта (например, "initConversation")

    Returns:
        Текст промпта из Langfuse, default_prompt (если передан и промпт не найден),
        или None, если промпт не найден и default_prompt не передан
    """
    current_time = time.time()
    source = "cache"
    calling_context = _get_calling_context()
    
    if use_cache and prompt_name in _PROMPT_CACHE:
        cached_prompt, cache_time = _PROMPT_CACHE[prompt_name]
        if current_time - cache_time < _PROMPT_CACHE_TTL:
            logger.debug(
                f"[PROMPT_USAGE] Промпт '{prompt_name}' использован из кэша",
                extra={
                    "prompt_name": prompt_name,
                    "source": "cache",
                    "label": langfuse_label,
                    "context": context,
                    "cache_age_seconds": int(current_time - cache_time),
                    **calling_context,
                }
            )
            return cached_prompt
        del _PROMPT_CACHE[prompt_name]
    
    langfuse_service = _get_langfuse_service()
    if langfuse_service and langfuse_service.is_enabled():
        try:
            prompt_text = langfuse_service.get_prompt_text(
                name=prompt_name,
                variables=variables,
                label=langfuse_label,
                version=None,
                fallback=None,
            )
            if prompt_text:
                source = "langfuse"
                if use_cache:
                    _PROMPT_CACHE[prompt_name] = (prompt_text, current_time)
                logger.info(
                    f"[PROMPT_USAGE] Промпт '{prompt_name}' загружен из Langfuse",
                    extra={
                        "prompt_name": prompt_name,
                        "source": "langfuse",
                        "label": langfuse_label,
                        "context": context,
                        "has_variables": bool(variables),
                        **calling_context,
                    }
                )
                return prompt_text
        except Exception as e:
            logger.debug(f"Не удалось загрузить промпт '{prompt_name}' из Langfuse: {e}")
    
    if default_prompt is not None:
        logger.warning(
            f"[PROMPT_USAGE] Промпт '{prompt_name}' не найден в Langfuse, используется дефолтный промпт",
            extra={
                "prompt_name": prompt_name,
                "source": "default",
                "label": langfuse_label,
                "context": context,
                **calling_context,
            }
        )
        return default_prompt
    
    logger.warning(
        f"[PROMPT_USAGE] Промпт '{prompt_name}' не найден в Langfuse и дефолтный промпт не передан",
        extra={
            "prompt_name": prompt_name,
            "source": "none",
            "label": langfuse_label,
            "context": context,
            **calling_context,
        }
    )
    return None


async def get_system_value(topic: str) -> Optional[str]:
    """Получает значение из таблицы myaso.system по topic.

    Args:
        topic: Название параметра системы

    Returns:
        Значение параметра или None, если параметр не найден
    """
    try:
        from src.services.database.utils import execute_with_timeout
        
        supabase = await get_supabase_client()

        result = await execute_with_timeout(
            supabase.table(TABLE_SYSTEM)
            .select(COLUMN_VALUE)
            .eq(COLUMN_TOPIC, topic)
            .execute(),
            operation_name=f"get_system_value({topic})",
        )

        if result.data and len(result.data) > 0:
            return result.data[0].get(COLUMN_VALUE)

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
        from src.services.database.utils import execute_with_timeout
        
        supabase = await get_supabase_client()

        result = await execute_with_timeout(
            supabase.table(TABLE_SYSTEM).select(f"{COLUMN_TOPIC}, {COLUMN_VALUE}").execute(),
            operation_name="get_all_system_values",
        )

        if result.data:
            return {row.get(COLUMN_TOPIC, ""): row.get(COLUMN_VALUE, "") for row in result.data}

        return {}
    except Exception as e:
        logger.error(f"Ошибка при получении всех значений системы: {e}")
        return {}


def escape_prompt_variables(prompt: str) -> str:
    """Экранирует переменные в промпте, которые не являются шаблонными переменными LangChain.

    LangChain create_agent API использует system_prompt напрямую, без шаблонных переменных.
    Однако для обратной совместимости и будущего использования оставляем проверку
    на старые переменные шаблона.

    Все фигурные скобки должны быть экранированы двойными скобками,
    чтобы они трактовались как буквальный текст, если это не системные переменные.

    Args:
        prompt: Текст промпта, который может содержать переменные в фигурных скобках

    Returns:
        Промпт с экранированными переменными
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


async def compose_prompts(
    prompt_names: List[str],
    separator: str = "\n\n",
    langfuse_label: Optional[str] = "production",
    variables: Optional[Dict[str, Any]] = None,
    context: Optional[str] = None,
) -> str:
    """Композирует несколько промптов в один.

    Args:
        prompt_names: Список названий промптов для композиции
        separator: Разделитель между промптами (по умолчанию "\n\n")
        langfuse_label: Метка версии промптов в Langfuse
        variables: Переменные для подстановки в промпты
        context: Дополнительный контекст использования (например, "initConversation", "processConversation")

    Returns:
        Объединенный текст всех промптов
    """
    start_time = time.time()
    calling_context = _get_calling_context()
    
    prompts = []
    loaded_prompts = []
    failed_prompts = []
    
    for prompt_name in prompt_names:
        if prompt_name:
            prompt_text = await get_prompt(
                prompt_name=prompt_name,
                default_prompt="",
                langfuse_label=langfuse_label,
                variables=variables,
                context=context,
            )
            if prompt_text:
                prompts.append(prompt_text)
                loaded_prompts.append(prompt_name)
            else:
                failed_prompts.append(prompt_name)
    
    composed_prompt = separator.join(prompts)
    duration_ms = int((time.time() - start_time) * 1000)
    
    logger.info(
        f"[PROMPT_COMPOSITION] Составлен промпт из {len(loaded_prompts)} компонентов",
        extra={
            "prompt_names": prompt_names,
            "loaded_prompts": loaded_prompts,
            "failed_prompts": failed_prompts,
            "total_prompts": len(prompt_names),
            "loaded_count": len(loaded_prompts),
            "failed_count": len(failed_prompts),
            "label": langfuse_label,
            "context": context,
            "duration_ms": duration_ms,
            "has_variables": bool(variables),
            **calling_context,
        }
    )
    
    return composed_prompt


