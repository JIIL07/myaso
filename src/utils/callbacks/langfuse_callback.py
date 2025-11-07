"""
LangfuseHandler - callback handler для Langfuse.

Отслеживает вызовы инструментов через LangFuse.
"""

from typing import Any, Dict, Optional
from langfuse import Langfuse
from langchain_core.callbacks.base import BaseCallbackHandler
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from src.config.settings import settings



class LangfuseHandler(BaseCallbackHandler):
    """
    LangFuse callback handler для отслеживания агентов.

    Отслеживает:
    - Вызовы инструментов
    """

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

        self._langfuse_handler: Optional[LangfuseCallbackHandler] = None
        self._langfuse_client: Optional[Any] = None

        if settings.langfuse.langfuse_enabled and settings.langfuse.langfuse_public_key:
            try:
                if Langfuse:
                    self._langfuse_client = Langfuse(
                        public_key=settings.langfuse.langfuse_public_key,
                        secret_key=settings.langfuse.langfuse_secret_key,
                        host=settings.langfuse.langfuse_host,
                    )

                self._langfuse_handler = LangfuseCallbackHandler(
                    public_key=settings.langfuse.langfuse_public_key,
                    secret_key=settings.langfuse.langfuse_secret_key,
                    host=settings.langfuse.langfuse_host,
                    user_id=client_phone,
                    **kwargs
                )
            except Exception:
                pass

        self._trace_id: Optional[str] = None
        self._run_manager: Optional[Any] = None


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

            if self._langfuse_handler:
                try:
                    self._langfuse_handler.on_chain_start(serialized, inputs, **kwargs)
                except Exception:
                    pass

            if 'run_manager' in kwargs:
                self._run_manager = kwargs['run_manager']
            self._update_trace_id(**kwargs)

        except Exception:
            pass

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
        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_start(serialized, input_str, **kwargs)
            except Exception:
                pass

        self._update_trace_id(**kwargs)

    def on_tool_end(
        self,
        output: str,
        **kwargs
    ):
        """Вызывается когда инструмент завершает выполнение."""
        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_end(output, **kwargs)
            except Exception:
                pass


    def on_tool_error(
        self,
        error: Exception,
        **kwargs
    ):
        """Вызывается когда инструмент встречает ошибку."""
        if self._langfuse_handler:
            try:
                self._langfuse_handler.on_tool_error(error, **kwargs)
            except Exception:
                pass


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
                except Exception:
                    pass

        except Exception:
            pass

    def save_conversation_to_langfuse(self) -> None:
        """Сохраняет информацию о разговоре и отправляет на cloud.langfuse."""
        if self._langfuse_client:
            try:
                self._langfuse_client.flush()
            except Exception:
                pass
        elif self._langfuse_handler:
            try:
                if hasattr(self._langfuse_handler, 'langfuse'):
                    langfuse_obj = getattr(self._langfuse_handler, 'langfuse')
                    if langfuse_obj and hasattr(langfuse_obj, 'flush'):
                        langfuse_obj.flush()
                elif hasattr(self._langfuse_handler, 'flush'):
                    self._langfuse_handler.flush()
            except Exception:
                pass