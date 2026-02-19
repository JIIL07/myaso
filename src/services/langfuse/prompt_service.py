from __future__ import annotations

import logging
from typing import Any, Optional, Union

from langfuse import get_client

from src.services.langfuse.config import LangFuseConfig

logger = logging.getLogger(__name__)


class LangfusePromptService:
    """Service for fetching and managing prompts from Langfuse."""

    def __init__(self, config: Optional[LangFuseConfig] = None) -> None:
        self.config = config
        self._client = None
        self._enabled = False

        if self.config and self.config.langfuse_enabled:
            try:
                if self.config.is_configured_instance():
                    from langfuse import Langfuse

                    self._client = Langfuse(
                        public_key=self.config.langfuse_public_key,
                        secret_key=self.config.langfuse_secret_key,
                        host=self.config.langfuse_host,
                    )

                    if self._client.auth_check():
                        self._enabled = True
                        logger.info("[Langfuse] Prompt service initialized and authenticated")
                    else:
                        logger.warning("[Langfuse] Authentication check failed")
                        self._client = None
                else:
                    logger.warning(
                        "Langfuse not configured. Prompts will be loaded from database."
                    )
            except Exception as e:
                logger.error("[Langfuse] Init error: %s", e, exc_info=True)
                self._enabled = False
                self._client = None

    def is_enabled(self) -> bool:
        """Check whether the Langfuse service is enabled."""
        return self._enabled and self._client is not None

    def get_prompt(
        self,
        name: str,
        label: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Optional[Any]:
        """Fetch a prompt from Langfuse.

        Args:
            name: Prompt name in Langfuse.
            label: Label for filtering versions (e.g. "production").
            version: Specific prompt version (optional).

        Returns:
            Langfuse prompt object or None if not found / service disabled.
        """
        if not self.is_enabled():
            return None

        try:
            kwargs: dict[str, Any] = {"name": name}
            if label:
                kwargs["label"] = label
            if version is not None:
                kwargs["version"] = version

            prompt = self._client.get_prompt(**kwargs)
            logger.debug("[Langfuse] Prompt '%s' loaded", name)
            return prompt
        except Exception as e:
            logger.warning("[Langfuse] Failed to load prompt '%s': %s", name, e)
            return None

    def compile_prompt(
        self,
        prompt: Any,
        variables: Optional[dict[str, Any]] = None,
    ) -> Union[str, list[dict[str, str]]]:
        """Compile a prompt with variables.

        Args:
            prompt: Langfuse prompt object.
            variables: Dictionary of variables for substitution.

        Returns:
            For text prompts — a string, for chat prompts — a list of messages.
        """
        if not prompt:
            raise ValueError("Prompt cannot be None")

        try:
            if variables:
                return prompt.compile(**variables)
            return prompt.prompt if hasattr(prompt, "prompt") else prompt
        except Exception as e:
            logger.error("[Langfuse] Prompt compile error: %s", e)
            raise

    def get_prompt_config(self, prompt: Any) -> dict[str, Any]:
        """Get prompt configuration (model, temperature, etc.)."""
        if not prompt:
            return {}
        try:
            if hasattr(prompt, "config"):
                return prompt.config if isinstance(prompt.config, dict) else {}
            return {}
        except Exception as e:
            logger.warning("[Langfuse] Failed to get prompt config: %s", e)
            return {}

    def create_prompt(
        self,
        name: str,
        prompt: Union[str, list[dict[str, str]]],
        type: str = "text",
        labels: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Create a prompt in Langfuse."""
        if not self.is_enabled():
            logger.warning("[Langfuse] Not enabled, cannot create prompt")
            return False

        try:
            request_data: dict[str, Any] = {
                "name": name,
                "prompt": prompt,
                "type": type,
            }
            if labels:
                request_data["labels"] = labels
            if config:
                request_data["config"] = config
            if tags:
                request_data["tags"] = tags

            self._client.api.prompts.create(request=request_data)
            logger.info("[Langfuse] Prompt '%s' created", name)
            return True
        except Exception as e:
            logger.error("[Langfuse] Error creating prompt '%s': %s", name, e)
            return False

    def get_prompt_text(
        self,
        name: str,
        variables: Optional[dict[str, Any]] = None,
        label: Optional[str] = None,
        version: Optional[int] = None,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        """Fetch and compile a text prompt from Langfuse."""
        prompt = self.get_prompt(name=name, label=label, version=version)
        if not prompt:
            return fallback

        try:
            prompt_type = getattr(prompt, "type", None)
            if prompt_type is None:
                class_name = type(prompt).__name__
                if "Chat" in class_name:
                    logger.warning("[Langfuse] Prompt '%s' is chat type, using fallback", name)
                    return fallback
            elif prompt_type != "text":
                logger.warning("[Langfuse] Prompt '%s' type=%s, using fallback", name, prompt_type)
                return fallback

            compiled = self.compile_prompt(prompt, variables)
            if isinstance(compiled, str):
                return compiled
            logger.warning("[Langfuse] Prompt '%s' is not text, using fallback", name)
            return fallback
        except Exception as e:
            logger.error("[Langfuse] Compile error for '%s': %s", name, e)
            return fallback

    def get_prompt_chat(
        self,
        name: str,
        variables: Optional[dict[str, Any]] = None,
        label: Optional[str] = None,
        version: Optional[int] = None,
        fallback: Optional[list[dict[str, str]]] = None,
    ) -> Optional[list[dict[str, str]]]:
        """Fetch and compile a chat prompt from Langfuse."""
        prompt = self.get_prompt(name=name, label=label, version=version)
        if not prompt:
            return fallback

        try:
            prompt_type = getattr(prompt, "type", None)
            if prompt_type is None:
                class_name = type(prompt).__name__
                if "Text" in class_name and "Chat" not in class_name:
                    logger.warning("[Langfuse] Prompt '%s' is text type, using fallback", name)
                    return fallback
            elif prompt_type != "chat":
                logger.warning("[Langfuse] Prompt '%s' type=%s, using fallback", name, prompt_type)
                return fallback

            compiled = self.compile_prompt(prompt, variables)
            if isinstance(compiled, list):
                return compiled
            logger.warning("[Langfuse] Prompt '%s' is not chat, using fallback", name)
            return fallback
        except Exception as e:
            logger.error("[Langfuse] Compile error for '%s': %s", name, e)
            return fallback
