import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from src.config.settings import settings
from src.constants import DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        if not hasattr(settings, "openrouter"):
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
        if self._llm is None:
            try:
                self._llm = ChatOpenAI(
                    model=self.model_id,
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=self.temperature,
                    default_headers={
                        "X-Title": "Myaso AI Agent",
                    },
                )
            except Exception as e:
                raise ValueError(f"Не удалось инициализировать LLM: {e}") from e
        return self._llm

    def reset(self) -> None:
        self._llm = None
