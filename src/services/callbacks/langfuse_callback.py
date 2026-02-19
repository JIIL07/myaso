import logging
import os
from typing import Any, Optional

from langfuse import observe, propagate_attributes, Langfuse
from langfuse.langchain import CallbackHandler

from src.config.settings import settings

logger = logging.getLogger(__name__)

_env_vars_set = False
_langfuse_client: Optional[Langfuse] = None


def is_langfuse_enabled() -> bool:
    return bool(
        settings.langfuse.langfuse_enabled
        and settings.langfuse.langfuse_public_key
        and settings.langfuse.langfuse_secret_key
    )


def _ensure_langfuse_client() -> Optional[Langfuse]:
    global _langfuse_client, _env_vars_set

    if _langfuse_client is not None:
        return _langfuse_client

    if not is_langfuse_enabled():
        return None

    try:
        if not _env_vars_set:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse.langfuse_public_key
            os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse.langfuse_secret_key
            os.environ["LANGFUSE_HOST"] = settings.langfuse.langfuse_host
            _env_vars_set = True
            logger.debug("[Langfuse] Environment variables set")

        _langfuse_client = Langfuse(
            public_key=settings.langfuse.langfuse_public_key,
            secret_key=settings.langfuse.langfuse_secret_key,
            host=settings.langfuse.langfuse_host,
            flush_interval=settings.langfuse.langfuse_flush_interval,
        )

        if _langfuse_client.auth_check():
            logger.info("[Langfuse] Client initialized and authenticated")
            return _langfuse_client
        else:
            logger.warning("[Langfuse] Authentication check failed")
            _langfuse_client = None
            return None
    except Exception as e:
        logger.error("[Langfuse] Failed to initialize client: %s", e, exc_info=True)
        _langfuse_client = None
        return None


def create_langfuse_callback_handler() -> Optional[CallbackHandler]:
    if not is_langfuse_enabled():
        return None

    client = _ensure_langfuse_client()
    if not client:
        logger.warning("[Langfuse] Cannot create CallbackHandler: client not initialized")
        return None

    try:
        handler = CallbackHandler(
            public_key=settings.langfuse.langfuse_public_key,
        )
        logger.info("[Langfuse] CallbackHandler created successfully")
        return handler
    except Exception as e:
        logger.warning("[Langfuse] Failed to create CallbackHandler: %s", e, exc_info=True)
        return None


def flush_langfuse() -> None:
    if not is_langfuse_enabled():
        return

    client = _ensure_langfuse_client()
    if not client:
        return

    try:
        client.flush()
        logger.debug("[Langfuse] Flushed traces successfully")
    except Exception as e:
        logger.warning("[Langfuse] Error during flush: %s", e, exc_info=True)


def update_trace(
    *,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Update current trace with metadata."""
    if not is_langfuse_enabled():
        return

    client = _ensure_langfuse_client()
    if not client:
        logger.debug("[Langfuse] Client not available for trace update")
        return

    try:

        kwargs: dict[str, Any] = {}
        if name is not None:
            kwargs["name"] = name
        if user_id is not None:
            kwargs["user_id"] = user_id
        if session_id is not None:
            kwargs["session_id"] = session_id
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if tags is not None:
            kwargs["tags"] = tags
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            client.update_current_trace(**kwargs)
            logger.debug("[Langfuse] Trace updated: %s", list(kwargs.keys()))
    except Exception as e:
        logger.debug("[Langfuse] Could not update trace: %s", e)
