"""
LangfuseHandler - упрощенный callback handler для Langfuse.

Отслеживает вызовы инструментов и обновляет метаданные трейсов.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
from langchain_core.callbacks.base import BaseCallbackHandler
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from src.config.settings import settings

logger = logging.getLogger(__name__)


class LangfuseHandler(BaseCallbackHandler):
    """
    Упрощенный LangFuse callback handler для отслеживания агентов.

    Отслеживает:
    - Вызовы инструментов
    - Метаданные трейсов (tools_used)
    """

    TOOL_TYPE_MAP = {
        "enhance_user_product_query": "[VECTOR SEARCH]",
        "text_to_sql_products": "[TEXT-TO-SQL]",
        "generate_sql_from_text": "[SQL GENERATOR]",
        "execute_sql_conditions": "[SQL EXECUTOR]",
        "show_product_photos": "[PHOTO SENDER]",
        "get_client_profile": "[CLIENT PROFILE]",
        "get_client_orders": "[CLIENT ORDERS]",
        "get_random_products": "[RANDOM PRODUCTS]",
    }

    def __init__(
        self,
        client_phone: str,
        session_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        **kwargs
    ):
        """Инициализация LangfuseHandler.

        Args:
            client_phone: Номер телефона клиента
            session_id: ID сессии (опционально)
            trace_name: Имя трейса (по умолчанию "AgentExecutor")
            **kwargs: Дополнительные параметры для CallbackHandler
        """
        super().__init__()

        self.client_phone = client_phone
        self.trace_name = trace_name or "AgentExecutor"

        logger.info(
            f"[LangfuseHandler.__init__] Инициализация для {client_phone}, "
            f"langfuse_enabled={settings.langfuse.langfuse_enabled}, "
            f"has_public_key={bool(settings.langfuse.langfuse_public_key)}"
        )

        self._langfuse_handler: Optional[LangfuseCallbackHandler] = None
        if settings.langfuse.langfuse_enabled and settings.langfuse.langfuse_public_key:
            try:
                self._langfuse_handler = LangfuseCallbackHandler(
                    public_key=settings.langfuse.langfuse_public_key,
                    secret_key=settings.langfuse.langfuse_secret_key,
                    host=settings.langfuse.langfuse_host,
                    user_id=client_phone,
                    **kwargs
                )
                logger.info(f"LangFuse CallbackHandler инициализирован для {client_phone}")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать LangFuse CallbackHandler: {e}", exc_info=True)
        else:
            logger.warning(f"Langfuse отключен или нет ключей для {client_phone}")

        self.used_tools: set = set()
        self.tool_calls: List[Dict[str, Any]] = []
        self._trace_id: Optional[str] = None
        self._run_manager: Optional[Any] = None

        logger.info(
            f"[LangfuseHandler.__init__] Завершена инициализация для {client_phone}, "
            f"type={type(self).__name__}, has_langfuse={self._langfuse_handler is not None}"
        )

    def _get_tool_type(self, tool_name: str) -> str:
        """Возвращает тип инструмента для логирования."""
        return self.TOOL_TYPE_MAP.get(tool_name, "[TOOL]")

    def _update_trace_id(self, **kwargs) -> None:
        """Обновляет trace_id из доступных источников."""
        if self._trace_id:
            return

        if 'run_manager' in kwargs:
            run_manager = kwargs['run_manager']
            if run_manager:
                if hasattr(run_manager, 'parent_run_id') and run_manager.parent_run_id:
                    self._trace_id = run_manager.parent_run_id
                elif hasattr(run_manager, 'run_id') and run_manager.run_id:
                    self._trace_id = run_manager.run_id

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Вызывается когда chain начинает выполнение."""
        try:
            serialized_type = type(serialized).__name__
            serialized_str = str(serialized)[:200] if serialized else "None"
            chain_name = serialized.get('name', 'unknown') if isinstance(serialized, dict) else 'unknown'

            logger.info(
                f"[LangfuseHandler.on_chain_start] Chain '{chain_name}' начал выполнение для {self.client_phone}",
                extra={
                    "chain_name": chain_name,
                    "client_phone": self.client_phone,
                    "has_run_manager": 'run_manager' in kwargs,
                    "serialized_type": serialized_type,
                    "serialized_preview": serialized_str
                }
            )

            if self._is_root_chain(serialized):
                serialized = self._modify_chain_name(serialized)

            self.used_tools.clear()
            self.tool_calls = []

            if self._langfuse_handler:
                try:
                    self._langfuse_handler.on_chain_start(serialized, inputs, **kwargs)
                except Exception as e:
                    logger.warning(f"Ошибка в LangFuse on_chain_start: {e}", exc_info=True)

            if 'run_manager' in kwargs:
                self._run_manager = kwargs['run_manager']
                logger.debug(f"[LangfuseHandler] run_manager установлен для {self.client_phone}")
            self._update_trace_id(**kwargs)

        except Exception as e:
            logger.error(f"[LangfuseHandler] Ошибка в on_chain_start: {e}", exc_info=True)

    def _is_root_chain(self, serialized: Any) -> bool:
        """Проверяет, является ли chain корневым AgentExecutor."""
        if not serialized or not isinstance(serialized, dict):
            return False
        return serialized.get('name') == "AgentExecutor"

    def _modify_chain_name(self, serialized: Dict[str, Any]) -> Dict[str, Any]:
        """Изменяет имя chain на trace_name."""
        modified = dict(serialized)
        modified['name'] = self.trace_name
        logger.debug(f"[LangfuseHandler] Изменено имя chain на '{self.trace_name}'")
        return modified

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ):
        """Вызывается когда инструмент начинает выполнение."""
        tool_name = serialized.get("name", "unknown_tool") if isinstance(serialized, dict) else "unknown_tool"

        print(f"🔧 TOOL START: {tool_name}")

        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_start(serialized, input_str, **kwargs)
            except Exception as e:
                logger.warning(
                    f"[LangfuseHandler] Ошибка в LangFuse on_tool_start: {e}",
                    exc_info=True
                )

        logger.info(
            f"[LangfuseHandler.on_tool_start] ВЫЗВАН для tool='{tool_name}', client={self.client_phone}",
            extra={
                "tool_name": tool_name,
                "client_phone": self.client_phone,
                "serialized_type": type(serialized).__name__,
                "has_run_manager": 'run_manager' in kwargs,
                "trace_id": self._trace_id
            }
        )

        self.used_tools.add(tool_name)
        tool_type = self._get_tool_type(tool_name)

        self._update_trace_id(**kwargs)

        logger.info(
            f"[TOOL CALL] {tool_type} '{tool_name}' вызван для {self.client_phone}",
            extra={
                "tool_name": tool_name,
                "tool_type": tool_type,
                "client_phone": self.client_phone,
                "trace_id": self._trace_id
            }
        )

        self.tool_calls.append({
            "tool_name": tool_name,
            "input": input_str,
            "start_time": datetime.now().isoformat(),
            "output": None,
            "error": None
        })

    def on_tool_end(
        self,
        output: str,
        **kwargs
    ):
        """Вызывается когда инструмент завершает выполнение."""
        if not self.tool_calls:
            return

        tool_call = self.tool_calls[-1]
        tool_name = tool_call["tool_name"]

        print(f"✅ TOOL END: {tool_name}")

        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_end(output, **kwargs)
            except Exception as e:
                logger.warning(
                    f"[LangfuseHandler] Ошибка в LangFuse on_tool_end: {e}",
                    exc_info=True
                )

        tool_call["output"] = output
        tool_call["end_time"] = datetime.now().isoformat()

        tool_type = self._get_tool_type(tool_name)
        logger.info(
            f"[TOOL END] {tool_type} '{tool_name}' завершен для {self.client_phone}",
            extra={
                "tool_name": tool_name,
                "tool_type": tool_type,
                "client_phone": self.client_phone,
                "trace_id": self._trace_id
            }
        )

    def on_tool_error(
        self,
        error: Exception,
        **kwargs
    ):
        """Вызывается когда инструмент встречает ошибку."""
        if not self.tool_calls:
            return

        tool_call = self.tool_calls[-1]
        tool_name = tool_call["tool_name"]

        print(f"❌ TOOL ERROR: {tool_name} - {error}")

        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_error(error, **kwargs)
            except Exception as e:
                logger.warning(
                    f"[LangfuseHandler] Ошибка в LangFuse on_tool_error: {e}",
                    exc_info=True
                )

        tool_call["error"] = str(error)

        tool_type = self._get_tool_type(tool_name)
        logger.error(
            f"[TOOL ERROR] {tool_type} '{tool_name}' "
            f"завершился с ошибкой для {self.client_phone}",
            exc_info=True,
            extra={
                "tool_name": tool_name,
                "tool_type": tool_type,
                "error": str(error),
                "client_phone": self.client_phone,
                "trace_id": self._trace_id
            }
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Вызывается когда chain завершает выполнение."""
        try:
            self._log_used_tools()

            if self.used_tools:
                self._update_trace_metadata()

            if self._langfuse_handler:
                try:
                    self._langfuse_handler.on_chain_end(outputs, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"[LangfuseHandler] Ошибка в LangFuse on_chain_end: {e}",
                        exc_info=True
                    )

            if self.used_tools:
                self._update_trace_metadata()

        except Exception as e:
            logger.warning(f"[LangfuseHandler] Ошибка в on_chain_end: {e}", exc_info=True)

    def _log_used_tools(self) -> None:
        """Логирует информацию об использованных инструментах."""
        tools_list = sorted(list(self.used_tools))

        if not tools_list:
            logger.warning(
                f"[LangfuseHandler] tools_used пустой для {self.client_phone}! "
                f"tool_calls: {[tc.get('tool_name') for tc in self.tool_calls]}"
            )
            return

        tools_summary = []
        for tool_name in tools_list:
            call_count = sum(1 for tc in self.tool_calls if tc.get("tool_name") == tool_name)
            tool_type = self._get_tool_type(tool_name).replace("[", "").replace("]", "")
            tools_summary.append(f"{tool_type} {tool_name}({call_count}x)")

        logger.info(
            f"Использовано инструментов для {self.client_phone}: {', '.join(tools_summary)}"
        )

    def _update_trace_metadata(self) -> None:
        """Обновляет метаданные трейса со списком использованных инструментов."""
        tools_list = sorted(list(self.used_tools))
        if not tools_list:
            return

        if self._run_manager and hasattr(self._run_manager, 'get_parent_run'):
            try:
                parent_run = self._run_manager.get_parent_run()
                if parent_run and hasattr(parent_run, 'extra'):
                    if parent_run.extra is None:
                        parent_run.extra = {}
                    parent_run.extra['tools_used'] = tools_list
                    logger.debug(f"[LangfuseHandler] Обновлены метаданные через run_manager: {tools_list}")
            except Exception as e:
                logger.debug(f"[LangfuseHandler] Не удалось обновить через run_manager: {e}")

    def save_conversation_to_langfuse(self) -> None:
        """Сохраняет информацию о разговоре (для совместимости)."""
        if self.used_tools:
            self._update_trace_metadata()
