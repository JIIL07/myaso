import logging
import re
import time
from typing import Any, Optional

from src.constants import (
    COLUMN_TOPIC,
    COLUMN_VALUE,
    TABLE_SYSTEM,
    PROMPT_CACHE_TTL,
)
from src.services.database.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

_PROMPT_CACHE: dict[str, tuple[str, float]] = {}
_langfuse_prompt_service: Optional[Any] = None
_langfuse_lock = None


def _get_langfuse_service() -> Optional[Any]:
    global _langfuse_prompt_service, _langfuse_lock

    if _langfuse_prompt_service is not None:
        return _langfuse_prompt_service

    if _langfuse_lock is None:
        import threading
        _langfuse_lock = threading.Lock()

    with _langfuse_lock:
        if _langfuse_prompt_service is not None:
            return _langfuse_prompt_service
        try:
            from src.config.settings import settings
            from src.services.langfuse.prompt_service import LangfusePromptService
            if hasattr(settings, "langfuse"):
                _langfuse_prompt_service = LangfusePromptService(config=settings.langfuse)
                return _langfuse_prompt_service
        except Exception as e:
            logger.debug("[Prompt] Failed to initialize Langfuse: %s", e)

    return None


async def get_prompt(
    prompt_name: str,
    default_prompt: Optional[str] = None,
    use_cache: bool = True,
    langfuse_label: Optional[str] = "production",
    variables: Optional[dict[str, Any]] = None,
    context: Optional[str] = None,
) -> Optional[str]:
    current_time = time.time()

    if use_cache and prompt_name in _PROMPT_CACHE:
        cached_prompt, cache_time = _PROMPT_CACHE[prompt_name]
        if current_time - cache_time < PROMPT_CACHE_TTL:
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
                if use_cache:
                    _PROMPT_CACHE[prompt_name] = (prompt_text, current_time)
                return prompt_text
        except Exception as e:
            logger.debug("[Prompt] Failed to load '%s' from Langfuse: %s", prompt_name, e)

    if default_prompt is not None:
        logger.warning("[Prompt] '%s' not found, using default", prompt_name)
        return default_prompt

    logger.warning("[Prompt] '%s' not found and no default provided", prompt_name)
    return None


async def get_system_value(topic: str) -> Optional[str]:
    try:
        from src.services.database.utils import execute_with_timeout
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table(TABLE_SYSTEM)
            .select(COLUMN_VALUE)
            .eq(COLUMN_TOPIC, topic)
            .execute(),
            operation_name="get_system_value(%s)" % topic,
        )
        if result.data and len(result.data) > 0:
            return result.data[0].get(COLUMN_VALUE)
        return None
    except Exception as e:
        logger.error("[Prompt] Error getting system value '%s': %s", topic, e)
        return None


async def get_all_system_values() -> dict[str, str]:
    try:
        from src.services.database.utils import execute_with_timeout
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table(TABLE_SYSTEM).select("%s, %s" % (COLUMN_TOPIC, COLUMN_VALUE)).execute(),
            operation_name="get_all_system_values",
        )
        if result.data:
            return {row.get(COLUMN_TOPIC, ""): row.get(COLUMN_VALUE, "") for row in result.data}
        return {}
    except Exception as e:
        logger.error("[Prompt] Error getting system values: %s", e)
        return {}


def escape_prompt_variables(prompt: str) -> str:
    known_variables = {"input", "chat_history", "agent_scratchpad", "intermediate_steps"}
    pattern = r"(?<!\{)\{([^}]+)\}(?!\})"

    def replace_var(match):
        var_name = match.group(1).strip()
        if var_name in known_variables:
            return match.group(0)
        return "{{%s}}" % var_name

    return re.sub(pattern, replace_var, prompt)


async def compose_prompts(
    prompt_names: list[str],
    separator: str = "\n\n",
    langfuse_label: Optional[str] = "production",
    variables: Optional[dict[str, Any]] = None,
    context: Optional[str] = None,
) -> str:
    prompts = []
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

    return separator.join(prompts)
