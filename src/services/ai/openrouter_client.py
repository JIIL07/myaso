"""Клиент для работы с OpenRouter API."""
import logging
from typing import Optional, Any

from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.services.ai.constants import DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Клиент для работы с OpenRouter API через LangChain ChatOpenAI."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        """Инициализирует OpenRouter клиент.

        Args:
            model_id: ID модели (если None, берется из settings)
            api_key: API ключ (если None, берется из settings)
            base_url: Базовый URL (если None, берется из settings)
            temperature: Температура модели (по умолчанию из constants)
        """
        if not hasattr(settings, 'openrouter'):
            raise ValueError("settings.openrouter не найден")

        self.model_id = model_id or settings.openrouter.model_id
        self.api_key = api_key or settings.openrouter.openrouter_api_key
        self.base_url = base_url or settings.openrouter.base_url
        self.temperature = temperature

        if not self.model_id:
            raise ValueError("model_id не установлен")
        
        if not self.api_key:
            raise ValueError("openrouter_api_key не установлен")

        self._llm: Optional[ChatOpenAI] = None

    def get_llm(self) -> ChatOpenAI:
        """Получает или создает ChatOpenAI клиент для OpenRouter.

        Returns:
            ChatOpenAI клиент, настроенный для OpenRouter
        """
        if self._llm is None:
            try:
                self._llm = ChatOpenAI(
                    model=self.model_id,
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=self.temperature,
                )
                logger.info(
                    f"[OpenRouterClient] Инициализирован клиент для модели {self.model_id}"
                )
            except Exception as e:
                logger.error(
                    f"[OpenRouterClient] Ошибка инициализации LLM: {e}",
                    exc_info=True
                )
                raise ValueError(f"Не удалось инициализировать LLM: {e}") from e

        return self._llm

    def reset(self) -> None:
        """Сбрасывает клиент (для пересоздания при необходимости)."""
        self._llm = None
        logger.debug("[OpenRouterClient] Клиент сброшен")
