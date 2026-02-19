import logging
from collections import deque
from datetime import datetime
from typing import Any, Optional

from langchain_core.callbacks.base import BaseCallbackHandler

from src.constants import MAX_LOGS

logger = logging.getLogger(__name__)


class AgentLoggerCallback(BaseCallbackHandler):
    def __init__(self, client_phone: str, logger_service: "AgentLogger") -> None:
        super().__init__()
        self.client_phone = client_phone
        self.logger_service = logger_service
        self._current_chain: Optional[str] = None
        self._current_tool: Optional[str] = None

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        try:
            if not serialized or not isinstance(serialized, dict):
                chain_name = "unknown"
            else:
                chain_name = serialized.get("name") or "unknown"
                if chain_name == "unknown":
                    chain_id = serialized.get("id")
                    if isinstance(chain_id, list) and chain_id:
                        chain_name = chain_id[-1]
                    elif isinstance(chain_id, str):
                        chain_name = chain_id

            self._current_chain = chain_name
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="chain_start",
                chain_name=chain_name,
                inputs=str(inputs)[:500] if inputs else None,
            )
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_chain_start: %s", e, exc_info=True)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="chain_end",
                chain_name=self._current_chain or "unknown",
                outputs=str(outputs)[:500] if outputs else None,
            )
            self._current_chain = None
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_chain_end: %s", e, exc_info=True)

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="chain_error",
                chain_name=self._current_chain or "unknown",
                error=str(error),
            )
            self._current_chain = None
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_chain_error: %s", e, exc_info=True)

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        try:
            tool_name = serialized.get("name", "unknown") if serialized and isinstance(serialized, dict) else "unknown"
            self._current_tool = tool_name
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="tool_start",
                tool_name=tool_name,
                tool_input=input_str[:500] if input_str else None,
            )
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_tool_start: %s", e, exc_info=True)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="tool_end",
                tool_name=self._current_tool or "unknown",
                tool_output=str(output)[:500] if output else None,
            )
            self._current_tool = None
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_tool_end: %s", e, exc_info=True)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="tool_error",
                tool_name=self._current_tool or "unknown",
                error=str(error),
            )
            self._current_tool = None
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_tool_error: %s", e, exc_info=True)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        try:
            safe_prompts = [p[:200] for p in prompts[:3]] if prompts and isinstance(prompts, list) else None
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="llm_start",
                prompts=safe_prompts,
            )
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_llm_start: %s", e, exc_info=True)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="llm_end",
                llm_response=str(response)[:500] if response else None,
            )
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_llm_end: %s", e, exc_info=True)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="llm_error",
                error=str(error),
            )
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_llm_error: %s", e, exc_info=True)

    def on_agent_action(self, action: Any, **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="agent_action",
                action=str(action)[:500] if action else None,
            )
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_agent_action: %s", e, exc_info=True)

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="agent_finish",
                finish=str(finish)[:500] if finish else None,
            )
        except Exception as e:
            logger.error("[AgentLoggerCallback] Error in on_agent_finish: %s", e, exc_info=True)


class AgentLogger:
    def __init__(self, max_logs: int = MAX_LOGS) -> None:
        self.max_logs = max_logs
        self._logs: deque = deque(maxlen=max_logs)

    def add_log(self, client_phone: str, event_type: str, **kwargs: Any) -> None:
        self._logs.append({
            "timestamp": datetime.now().isoformat(),
            "client_phone": client_phone,
            "event_type": event_type,
            **kwargs,
        })

    def get_callback_handler(self, client_phone: str) -> AgentLoggerCallback:
        return AgentLoggerCallback(client_phone=client_phone, logger_service=self)


_agent_logger_instance: Optional[AgentLogger] = None


def get_agent_logger() -> AgentLogger:
    global _agent_logger_instance
    if _agent_logger_instance is None:
        _agent_logger_instance = AgentLogger()
    return _agent_logger_instance
