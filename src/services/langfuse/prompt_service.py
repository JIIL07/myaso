"""Сервис для работы с промптами из Langfuse."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None

from src.services.langfuse.config import LangFuseConfig

logger = logging.getLogger(__name__)


class LangfusePromptService:
    """Сервис для получения и управления промптами из Langfuse."""

    def __init__(self, config: Optional[LangFuseConfig] = None):
        """Инициализация сервиса.

        Args:
            config: Конфигурация Langfuse. Если не указана, будет использована из settings.
        """
        self.config = config
        self._client: Optional[Langfuse] = None
        self._enabled = False

        if self.config and self.config.langfuse_enabled:
            try:
                if Langfuse is None:
                    logger.warning("Langfuse не установлен. Установите: pip install langfuse")
                    self._enabled = False
                elif self.config.is_configured_instance():
                    self._client = Langfuse(
                        public_key=self.config.langfuse_public_key,
                        secret_key=self.config.langfuse_secret_key,
                        host=self.config.langfuse_host,
                    )
                    self._enabled = True
                    logger.info("Langfuse prompt service initialized successfully")
                else:
                    logger.warning(
                        "Langfuse не настроен. Промпты будут загружаться из базы данных."
                    )
                    logger.debug(
                        f"Public key set: {bool(self.config.langfuse_public_key)}, "
                        f"Secret key set: {bool(self.config.langfuse_secret_key)}"
                    )
            except Exception as e:
                logger.error(f"Ошибка инициализации Langfuse: {e}")
                self._enabled = False

    def is_enabled(self) -> bool:
        """Проверяет, включен ли сервис Langfuse."""
        return self._enabled and self._client is not None

    def get_prompt(
        self,
        name: str,
        label: Optional[str] = None,
        version: Optional[int] = None,
    ) -> Optional[Any]:
        """Получает промпт из Langfuse.

        Args:
            name: Имя промпта в Langfuse
            label: Метка для фильтрации версий (например, "production")
            version: Конкретная версия промпта (опционально)

        Returns:
            Объект промпта из Langfuse или None, если не найден или сервис отключен
        """
        if not self.is_enabled():
            return None

        try:
            kwargs = {"name": name}
            if label:
                kwargs["label"] = label
            if version is not None:
                kwargs["version"] = version
            
            prompt = self._client.get_prompt(**kwargs)
            logger.debug(f"Промпт '{name}' успешно загружен из Langfuse")
            return prompt
        except Exception as e:
            logger.warning(
                f"Не удалось загрузить промпт '{name}' из Langfuse: {e}"
            )
            return None

    def compile_prompt(
        self,
        prompt: Any,
        variables: Optional[Dict[str, Any]] = None,
    ) -> Union[str, List[Dict[str, str]]]:
        """Компилирует промпт с переменными.

        Args:
            prompt: Объект промпта из Langfuse
            variables: Словарь переменных для подстановки

        Returns:
            Для text промптов - строка, для chat промптов - список сообщений
        """
        if not prompt:
            raise ValueError("Промпт не может быть None")

        try:
            if variables:
                compiled = prompt.compile(**variables)
            else:
                compiled = prompt.prompt if hasattr(prompt, "prompt") else prompt

            return compiled
        except Exception as e:
            logger.error(f"Ошибка компиляции промпта: {e}")
            raise

    def get_prompt_config(self, prompt: Any) -> Dict[str, Any]:
        """Получает конфигурацию промпта (model, temperature и т.д.).

        Args:
            prompt: Объект промпта из Langfuse

        Returns:
            Словарь с конфигурацией промпта
        """
        if not prompt:
            return {}

        try:
            if hasattr(prompt, "config"):
                return prompt.config if isinstance(prompt.config, dict) else {}
            return {}
        except Exception as e:
            logger.warning(f"Не удалось получить конфигурацию промпта: {e}")
            return {}

    def create_prompt(
        self,
        name: str,
        prompt: Union[str, List[Dict[str, str]]],
        type: str = "text",
        labels: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Создает промпт в Langfuse.

        Args:
            name: Имя промпта
            prompt: Текст промпта (для text) или список сообщений (для chat)
            type: Тип промпта ("text" или "chat")
            labels: Метки для промпта (например, ["production"])
            config: Конфигурация промпта (model, temperature и т.д.)
            tags: Теги для промпта

        Returns:
            True если промпт успешно создан, False в противном случае
        """
        if not self.is_enabled():
            logger.warning("Langfuse не включен, невозможно создать промпт")
            return False

        try:
            request_data = {
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
            logger.info(f"Промпт '{name}' успешно создан в Langfuse")
            return True
        except Exception as e:
            logger.error(f"Ошибка создания промпта '{name}' в Langfuse: {e}")
            return False

    def get_prompt_text(
        self,
        name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        version: Optional[int] = None,
        fallback: Optional[str] = None,
    ) -> Optional[str]:
        """Получает текстовый промпт из Langfuse и компилирует его с переменными.

        Args:
            name: Имя промпта в Langfuse
            variables: Переменные для подстановки в промпт
            label: Метка для фильтрации версий (например, "production")
            version: Конкретная версия промпта
            fallback: Значение по умолчанию, если промпт не найден

        Returns:
            Скомпилированный текстовый промпт или fallback
        """
        prompt = self.get_prompt(name=name, label=label, version=version)
        if not prompt:
            return fallback

        try:
            prompt_type = getattr(prompt, "type", None)
            if prompt_type is None:
                class_name = type(prompt).__name__
                if 'Chat' in class_name:
                    logger.warning(
                        f"Промпт '{name}' является chat промптом, возвращаю fallback"
                    )
                    return fallback
            elif prompt_type != "text":
                logger.warning(
                    f"Промпт '{name}' не является текстовым (type={prompt_type}), возвращаю fallback"
                )
                return fallback
            
            compiled = self.compile_prompt(prompt, variables)
            if isinstance(compiled, str):
                return compiled
            else:
                logger.warning(
                    f"Промпт '{name}' не является текстовым, возвращаю fallback"
                )
                return fallback
        except Exception as e:
            logger.error(f"Ошибка компиляции промпта '{name}': {e}")
            return fallback

    def get_prompt_chat(
        self,
        name: str,
        variables: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        version: Optional[int] = None,
        fallback: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """Получает chat промпт из Langfuse и компилирует его с переменными.

        Args:
            name: Имя промпта в Langfuse
            variables: Переменные для подстановки в промпт
            label: Метка для фильтрации версий (например, "production")
            version: Конкретная версия промпта
            fallback: Значение по умолчанию, если промпт не найден

        Returns:
            Скомпилированный chat промпт (список сообщений) или fallback
        """
        prompt = self.get_prompt(name=name, label=label, version=version)
        if not prompt:
            return fallback

        try:
            prompt_type = getattr(prompt, "type", None)
            if prompt_type is None:
                class_name = type(prompt).__name__
                if 'Text' in class_name and 'Chat' not in class_name:
                    logger.warning(
                        f"Промпт '{name}' является text промптом, возвращаю fallback"
                    )
                    return fallback
            elif prompt_type != "chat":
                logger.warning(
                    f"Промпт '{name}' не является chat промптом (type={prompt_type}), возвращаю fallback"
                )
                return fallback
            
            compiled = self.compile_prompt(prompt, variables)
            if isinstance(compiled, list):
                return compiled
            else:
                logger.warning(
                    f"Промпт '{name}' не является chat промптом, возвращаю fallback"
                )
                return fallback
        except Exception as e:
            logger.error(f"Ошибка компиляции промпта '{name}': {e}")
            return fallback
