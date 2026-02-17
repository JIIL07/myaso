"""Сервис для сбора и хранения логов агента."""
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Максимальное количество логов в памяти
MAX_LOGS = 500


class AgentLoggerCallback(BaseCallbackHandler):
    """Callback handler для сбора логов агента."""

    def __init__(self, client_phone: str, logger_service: "AgentLogger"):
        """Инициализирует callback handler.

        Args:
            client_phone: Номер телефона клиента
            logger_service: Сервис логирования
        """
        super().__init__()
        self.client_phone = client_phone
        self.logger_service = logger_service
        self._current_chain: Optional[str] = None
        self._current_tool: Optional[str] = None

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Вызывается когда chain начинает выполнение."""
        try:
            chain_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
            self._current_chain = chain_name

            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="chain_start",
                chain_name=chain_name,
                inputs=str(inputs)[:500] if inputs else None,
            )
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_chain_start: {e}", exc_info=True)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Вызывается когда chain завершает выполнение."""
        try:
            chain_name = self._current_chain or "unknown"
            
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="chain_end",
                chain_name=chain_name,
                outputs=str(outputs)[:500] if outputs else None,
            )
            self._current_chain = None
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_chain_end: {e}", exc_info=True)

    def on_chain_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда chain встречает ошибку."""
        try:
            chain_name = self._current_chain or "unknown"
            
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="chain_error",
                chain_name=chain_name,
                error=str(error),
            )
            self._current_chain = None
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_chain_error: {e}", exc_info=True)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда инструмент начинает выполнение."""
        try:
            tool_name = serialized.get("name", "unknown")
            self._current_tool = tool_name

            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="tool_start",
                tool_name=tool_name,
                tool_input=input_str[:500] if input_str else None,
            )
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_tool_start: {e}", exc_info=True)

    def on_tool_end(
        self,
        output: str,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда инструмент завершает выполнение."""
        try:
            tool_name = self._current_tool or "unknown"
            
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="tool_end",
                tool_name=tool_name,
                tool_output=str(output)[:500] if output else None,
            )
            self._current_tool = None
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_tool_end: {e}", exc_info=True)

    def on_tool_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда инструмент встречает ошибку."""
        try:
            tool_name = self._current_tool or "unknown"
            
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="tool_error",
                tool_name=tool_name,
                error=str(error),
            )
            self._current_tool = None
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_tool_error: {e}", exc_info=True)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Вызывается когда LLM начинает выполнение."""
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="llm_start",
                prompts=[p[:200] for p in prompts[:3]] if prompts else None,  # Первые 3 промпта, по 200 символов
            )
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_llm_start: {e}", exc_info=True)

    def on_llm_end(
        self,
        response: Any,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда LLM завершает выполнение."""
        try:
            response_str = str(response)[:500] if response else None
            
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="llm_end",
                llm_response=response_str,
            )
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_llm_end: {e}", exc_info=True)

    def on_llm_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда LLM встречает ошибку."""
        try:
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="llm_error",
                error=str(error),
            )
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_llm_error: {e}", exc_info=True)

    def on_agent_action(
        self,
        action: Any,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда агент выполняет действие."""
        try:
            action_str = str(action)[:500] if action else None
            
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="agent_action",
                action=action_str,
            )
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_agent_action: {e}", exc_info=True)

    def on_agent_finish(
        self,
        finish: Any,
        **kwargs: Any,
    ) -> None:
        """Вызывается когда агент завершает работу."""
        try:
            finish_str = str(finish)[:500] if finish else None
            
            self.logger_service.add_log(
                client_phone=self.client_phone,
                event_type="agent_finish",
                finish=finish_str,
            )
        except Exception as e:
            logger.error(f"[AgentLoggerCallback] Ошибка в on_agent_finish: {e}", exc_info=True)


class AgentLogger:
    """Сервис для сбора и хранения логов агента."""

    def __init__(self, max_logs: int = MAX_LOGS):
        """Инициализирует сервис логирования.

        Args:
            max_logs: Максимальное количество логов в памяти
        """
        self.max_logs = max_logs
        self._logs: deque = deque(maxlen=max_logs)

    def add_log(
        self,
        client_phone: str,
        event_type: str,
        **kwargs: Any,
    ) -> None:
        """Добавляет лог в хранилище.

        Args:
            client_phone: Номер телефона клиента
            event_type: Тип события (chain_start, tool_start, llm_start и т.д.)
            **kwargs: Дополнительные данные события
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "client_phone": client_phone,
            "event_type": event_type,
            **kwargs,
        }
        
        self._logs.append(log_entry)
        logger.debug(
            f"[AgentLogger] Добавлен лог: {event_type} для {client_phone}, "
            f"всего логов: {len(self._logs)}"
        )

    def get_logs(
        self,
        client_phone: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Возвращает логи с возможностью фильтрации.

        Args:
            client_phone: Фильтр по номеру телефона (если None, все клиенты)
            event_type: Фильтр по типу события (если None, все типы)
            limit: Максимальное количество логов (если None, все)

        Returns:
            Список логов, отсортированных по времени (новые первыми)
        """
        logs = list(self._logs)
        
        # Фильтрация
        if client_phone:
            logs = [log for log in logs if log.get("client_phone") == client_phone]
        
        if event_type:
            logs = [log for log in logs if log.get("event_type") == event_type]
        
        # Сортировка по времени (новые первыми)
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Ограничение количества
        if limit:
            logs = logs[:limit]
        
        return logs

    def get_callback_handler(self, client_phone: str) -> AgentLoggerCallback:
        """Создает callback handler для агента.

        Args:
            client_phone: Номер телефона клиента

        Returns:
            Callback handler для использования в LangChain
        """
        return AgentLoggerCallback(client_phone=client_phone, logger_service=self)

    def clear_logs(self) -> None:
        """Очищает все логи."""
        self._logs.clear()
        logger.info("[AgentLogger] Все логи очищены")

    def get_logs_count(self) -> int:
        """Возвращает количество логов в хранилище.

        Returns:
            Количество логов
        """
        return len(self._logs)


# Глобальный экземпляр сервиса логирования
_agent_logger_instance: Optional[AgentLogger] = None


def get_agent_logger() -> AgentLogger:
    """Получает глобальный экземпляр AgentLogger.

    Returns:
        Экземпляр AgentLogger
    """
    global _agent_logger_instance
    if _agent_logger_instance is None:
        _agent_logger_instance = AgentLogger()
    return _agent_logger_instance
