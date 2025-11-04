from __future__ import annotations

from typing import Any, List

from .base_agent import BaseAgent


class ProductAgent(BaseAgent):
    """Заглушка агента для продуктовых задач.

    Требует реализации промпта, инструментов и метода `run`.
    """

    def run(self, user_input: str, **kwargs: Any) -> Any:
        raise NotImplementedError("ProductAgent.run is not implemented yet.")

    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        raise NotImplementedError("ProductAgent._build_prompt is not implemented yet.")

    def _create_tools(self) -> List[Any]:
        raise NotImplementedError("ProductAgent._create_tools is not implemented yet.")