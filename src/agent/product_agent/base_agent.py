from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):

    def __init__(
        self,
        *,
        model: Any | None = None,
        tools: list[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools or []
        self.config = config or {}

    @abstractmethod
    def run(self, user_input: str, **kwargs: Any) -> Any: ...

    @abstractmethod
    def _build_prompt(self, user_input: str, **kwargs: Any) -> str: ...

    @abstractmethod
    def _create_tools(self) -> list[Any]: ...
