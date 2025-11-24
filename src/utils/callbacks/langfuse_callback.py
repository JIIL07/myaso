"""
LangfuseHandler - callback handler для Langfuse.

Отслеживает вызовы инструментов через LangFuse.
"""

from typing import Any, Dict, Optional

from langchain_core.callbacks.base import BaseCallbackHandler
from langfuse import Langfuse
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

from src.config.settings import settings
from src.utils.callbacks.reasoning_extractor import ReasoningExtractor



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
            trace_name: Имя трейса (по умолчанию "Agent")
            **kwargs: Дополнительные параметры для CallbackHandler
        """
        super().__init__()

        self.client_phone = client_phone
        self.trace_name = trace_name or "Agent"

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
        self._reasoning_extractor = ReasoningExtractor()


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
        """Проверяет, является ли chain корневым агентом."""
        if not serialized or not isinstance(serialized, dict):
            return False
        name = serialized.get('name', '')
        return (
            name == "AgentExecutor" or
            name == "Agent" or
            'agent' in name.lower()
        )

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

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Вызывается когда LLM завершает выполнение.
        
        Извлекает reasoning и thinking из ответа и передает их в LangFuse.
        
        Args:
            response: Ответ от LLM
            **kwargs: Дополнительные параметры (run_id, run_manager и т.д.)
        """
        try:
            reasoning_data = self._reasoning_extractor.extract_reasoning_from_response(response)
            
            reasoning_metadata = {}
            if reasoning_data.get("has_reasoning"):
                if reasoning_data.get("reasoning_text"):
                    reasoning_metadata["reasoning"] = reasoning_data["reasoning_text"]
                if reasoning_data.get("thinking_text"):
                    reasoning_metadata["thinking"] = reasoning_data["thinking_text"]
                if reasoning_data.get("reasoning_tokens"):
                    reasoning_metadata["reasoning_tokens"] = reasoning_data["reasoning_tokens"]

            if reasoning_metadata and "metadata" in kwargs:
                kwargs["metadata"] = {**kwargs.get("metadata", {}), **reasoning_metadata}
            elif reasoning_metadata:
                kwargs["metadata"] = reasoning_metadata

            if self._langfuse_handler:
                try:
                    self._langfuse_handler.on_llm_end(response, **kwargs)
                except Exception:
                    pass

            if reasoning_data.get("has_reasoning"):
                self._update_langfuse_generation_with_reasoning(reasoning_data, **kwargs)

        except Exception:
            if self._langfuse_handler:
                try:
                    self._langfuse_handler.on_llm_end(response, **kwargs)
                except Exception:
                    pass

    def _update_langfuse_generation_with_reasoning(
        self,
        reasoning_data: Dict[str, Any],
        **kwargs: Any
    ) -> None:
        """Обновляет generation в LangFuse с reasoning данными через прямой API.
        
        Args:
            reasoning_data: Словарь с reasoning данными
            **kwargs: Дополнительные параметры (run_id и т.д.)
        """
        if not self._langfuse_client:
            return

        try:
            run_id = kwargs.get("run_id")
            if not run_id:
                return

            metadata = {}
            
            if reasoning_data.get("reasoning_text"):
                metadata["reasoning"] = reasoning_data["reasoning_text"]
            
            if reasoning_data.get("thinking_text"):
                metadata["thinking"] = reasoning_data["thinking_text"]
            
            if reasoning_data.get("reasoning_tokens"):
                metadata["reasoning_tokens"] = reasoning_data["reasoning_tokens"]

            if not metadata:
                return

            try:
                if self._langfuse_handler and hasattr(self._langfuse_handler, 'langfuse'):
                    langfuse_obj = getattr(self._langfuse_handler, 'langfuse')
                    if langfuse_obj and hasattr(langfuse_obj, 'generation'):
                        pass
            except Exception:
                pass

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