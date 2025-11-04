from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAgent(ABC):
    """Абстрактный базовый класс агентов на LangChain.

    Определяет общий интерфейс и жизненный цикл. Наследники реализуют
    сбор промпта, создание инструментов и метод `run`.
    """

    def __init__(
        self,
        *,
        model: Any | None = None,
        tools: Optional[List[Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Инициализация агента.

        - model: LLM/chain для генерации ответов (необязательно)
        - tools: список инструментов агента (необязательно)
        - config: произвольные настройки для наследников (необязательно)
        """
        self.model = model
        self.tools = tools or []
        self.config = config or {}

    @abstractmethod
    def run(self, user_input: str, **kwargs: Any) -> Any:
        """Запускает основной сценарий агента для входной строки.

        Наследники собирают промпт, при необходимости используют инструменты
        и вызывают модель. Возвращаемый тип зависит от реализации.
        """

    @abstractmethod
    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        """Собирает промпт для модели (системные сообщения, контекст, инструкции)."""

    @abstractmethod
    def _create_tools(self) -> List[Any]:
        """Создаёт и возвращает список инструментов, доступных агенту."""


