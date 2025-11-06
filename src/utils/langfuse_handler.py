"""
LangfuseHandler - callback handler для Langfuse.

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
    LangFuse callback handler для отслеживания агентов.

    Отслеживает:
    - Вызовы инструментов
    - Метаданные трейсов (tools_used, user_id, session_id)
    """

    TOOL_TYPE_MAP = {
        "vector_search": "[VECTOR SEARCH]",
        "generate_sql_from_text": "[SQL GENERATOR]",
        "execute_sql_request": "[SQL EXECUTOR]",
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
        self.session_id = session_id or f"{client_phone}_{datetime.now().date()}"

        self._langfuse_handler: Optional[LangfuseCallbackHandler] = None
        if settings.langfuse.langfuse_enabled and settings.langfuse.langfuse_public_key:
            try:
                self._langfuse_handler = LangfuseCallbackHandler(
                    public_key=settings.langfuse.langfuse_public_key,
                    secret_key=settings.langfuse.langfuse_secret_key,
                    host=settings.langfuse.langfuse_host,
                    user_id=client_phone,
                    session_id=self.session_id,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Не удалось инициализировать LangFuse CallbackHandler: {e}", exc_info=True)


        self.used_tools: set = set()
        self.tool_calls: List[Dict[str, Any]] = []
        self._trace_id: Optional[str] = None
        self._run_manager: Optional[Any] = None

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
        return modified

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ):
        """Вызывается когда инструмент начинает выполнение."""
        tool_name = serialized.get("name", "unknown_tool") if isinstance(serialized, dict) else "unknown_tool"

        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_start(serialized, input_str, **kwargs)
            except Exception as e:
                logger.warning(
                    f"[LangfuseHandler] Ошибка в LangFuse on_tool_start: {e}",
                    exc_info=True
                )

        self.used_tools.add(tool_name)
        tool_type = self._get_tool_type(tool_name)

        self._update_trace_id(**kwargs)

        params_preview = str(input_str)[:200] if input_str else "нет параметров"
        trace_id_info = f" (trace_id: {self._trace_id})" if self._trace_id else ""
        logger.info(
            f"TOOLS: {tool_type} '{tool_name}' вызван для {self.client_phone}{trace_id_info} с параметрами: {params_preview}"
        )

        import time
        self.tool_calls.append({
            "tool_name": tool_name,
            "input": input_str,
            "start_time": datetime.now().isoformat(),
            "start_timestamp": time.time(),
            "output": None,
            "error": None,
            "duration": None
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

        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_end(output, **kwargs)
            except Exception as e:
                logger.warning(
                    f"[LangfuseHandler] Ошибка в LangFuse on_tool_end: {e}",
                    exc_info=True
                )

        import time
        tool_call["output"] = output
        tool_call["end_time"] = datetime.now().isoformat()

        if "start_timestamp" in tool_call:
            duration = time.time() - tool_call["start_timestamp"]
            tool_call["duration"] = round(duration, 3)

        tool_type = self._get_tool_type(tool_name)
        output_preview = str(output)[:300] if output else "нет результата"
        trace_id_info = f" (trace_id: {self._trace_id})" if self._trace_id else ""
        logger.info(
            f"TOOLS: {tool_type} '{tool_name}' завершен для {self.client_phone}{trace_id_info}, "
            f"длительность: {tool_call.get('duration', 'N/A')}s, "
            f"результат: {output_preview}"
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
            f"ОШИБКА: {tool_type} '{tool_name}' завершился с ошибкой для {self.client_phone}: {str(error)}",
            exc_info=True
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Вызывается когда chain завершает выполнение."""
        try:
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

    def _update_trace_metadata(self) -> None:
        """Обновляет метаданные трейса со списком использованных инструментов."""
        tools_list = sorted(list(self.used_tools))
        if not tools_list:
            return

        if self._run_manager and hasattr(self._run_manager, 'get_parent_run'):
            try:
                parent_run = self._run_manager.get_parent_run()
                if parent_run:
                    if hasattr(parent_run, 'extra'):
                        if parent_run.extra is None:
                            parent_run.extra = {}
                        parent_run.extra['tools_used'] = tools_list

                    if hasattr(parent_run, 'metadata'):
                        if parent_run.metadata is None:
                            parent_run.metadata = {}
                        parent_run.metadata['tools_used'] = tools_list
            except Exception:
                pass

    def save_conversation_to_langfuse(self) -> None:
        """Сохраняет информацию о разговоре в Langfuse."""
        if self.used_tools:
            self._update_trace_metadata()

        if self._langfuse_handler:
            try:
                if hasattr(self._langfuse_handler, 'flush'):
                    self._langfuse_handler.flush()
            except Exception:
                pass
