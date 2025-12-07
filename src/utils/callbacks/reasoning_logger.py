"""ReasoningLogger - callback handler для логирования reasoning агента.

Логирует:
- Вызовы модели (LLM)
- Tool calls от модели
- Reasoning и промежуточные шаги агента
"""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage

logger = logging.getLogger(__name__)


class ReasoningLogger(BaseCallbackHandler):
    """Callback handler для логирования reasoning и tool calls агента."""

    def __init__(self, client_phone: str, **kwargs):
        """Инициализация ReasoningLogger.

        Args:
            client_phone: Номер телефона клиента для логирования
            **kwargs: Дополнительные параметры
        """
        super().__init__()
        self.client_phone = client_phone
        self._llm_calls: List[Dict[str, Any]] = []
        self._tool_calls: List[Dict[str, Any]] = []

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any
    ) -> None:
        """Вызывается когда LLM начинает выполнение."""
        pass

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Вызывается когда LLM встречает ошибку."""
        try:
            run_id = kwargs.get("run_id", "unknown")
            logger.error(
                f"[ReasoningLogger] LLM ERROR для {self.client_phone} (run_id={run_id}): {error}"
            )
        except Exception:
            pass

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any
    ) -> None:
        """Вызывается когда инструмент начинает выполнение."""
        try:
            tool_name = serialized.get("name", "unknown")
            run_id = kwargs.get("run_id", "unknown")
            self._tool_calls.append({
                "run_id": run_id,
                "tool_name": tool_name,
                "input": input_str,
                "status": "started",
            })
        except Exception:
            pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Вызывается когда инструмент завершает выполнение."""
        try:
            tool_name = kwargs.get("name", "unknown")
            run_id = kwargs.get("run_id", "unknown")
            
            for tool_call in self._tool_calls:
                if tool_call.get("run_id") == run_id:
                    tool_call["status"] = "completed"
                    tool_call["output"] = output[:500] if output else ""
                    break
        except Exception:
            pass

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Вызывается когда инструмент встречает ошибку."""
        try:
            tool_name = kwargs.get("name", "unknown")
            run_id = kwargs.get("run_id", "unknown")
            logger.error(
                f"[ReasoningLogger] TOOL ERROR для {self.client_phone}: "
                f"{tool_name} (run_id={run_id}): {error}"
            )
        except Exception:
            pass

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Вызывается когда chain начинает выполнение."""
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Вызывается когда chain завершает выполнение."""
        pass

    def get_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по reasoning."""
        return {
            "llm_calls": len(self._llm_calls),
            "tool_calls": len(self._tool_calls),
            "llm_calls_with_tools": sum(
                1 for call in self._llm_calls if call.get("tool_calls")
            ),
            "llm_calls_without_tools": sum(
                1 for call in self._llm_calls if not call.get("tool_calls")
            ),
        }

