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
        # Не логируем здесь, чтобы избежать дублирования
        pass

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Вызывается когда LLM завершает выполнение."""
        try:
            run_id = kwargs.get("run_id", "unknown")
            
            # Извлекаем информацию о tool calls и reasoning
            tool_calls_info = []
            content = ""
            reasoning_content = ""
            response_metadata = {}
            
            # Проверяем разные форматы ответа
            # Формат 1: response.generations[0][0].message (стандартный LangChain)
            if hasattr(response, "generations") and response.generations:
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, "message"):
                            message = generation.message
                            if isinstance(message, AIMessage):
                                if hasattr(message, "tool_calls") and message.tool_calls:
                                    tool_calls_info = message.tool_calls
                                if hasattr(message, "content"):
                                    content = message.content or ""
                                # Извлекаем reasoning из response_metadata
                                if hasattr(message, "response_metadata"):
                                    response_metadata = message.response_metadata or {}
            
            # Формат 2: response.llm_output
            if not tool_calls_info and hasattr(response, "llm_output"):
                llm_output = response.llm_output or {}
                if "tool_calls" in llm_output:
                    tool_calls_info = llm_output["tool_calls"]
            
            # Формат 3: response.message (прямой доступ)
            if not tool_calls_info and hasattr(response, "message"):
                message = response.message
                if isinstance(message, AIMessage):
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        tool_calls_info = message.tool_calls
                    if hasattr(message, "content"):
                        content = message.content or ""
                    # Извлекаем reasoning из response_metadata
                    if hasattr(message, "response_metadata"):
                        response_metadata = message.response_metadata or {}
            
            # Извлекаем reasoning из content_blocks (для моделей с reasoning)
            message_for_reasoning = None
            if hasattr(response, "generations") and response.generations:
                for generation_list in response.generations:
                    for generation in generation_list:
                        if hasattr(generation, "message") and isinstance(generation.message, AIMessage):
                            message_for_reasoning = generation.message
                            break
            elif hasattr(response, "message") and isinstance(response.message, AIMessage):
                message_for_reasoning = response.message
            
            if message_for_reasoning:
                # Проверяем content_blocks для reasoning (LangChain стандарт)
                if hasattr(message_for_reasoning, "content_blocks"):
                    try:
                        content_blocks = message_for_reasoning.content_blocks
                        if content_blocks:
                            reasoning_blocks = []
                            for block in content_blocks:
                                # Проверяем разные форматы блоков
                                block_type = None
                                reasoning_text = None
                                
                                if isinstance(block, dict):
                                    block_type = block.get("type")
                                    reasoning_text = block.get("reasoning") or block.get("thinking")
                                elif hasattr(block, "type"):
                                    block_type = getattr(block, "type", None)
                                    if hasattr(block, "reasoning"):
                                        reasoning_text = getattr(block, "reasoning", None)
                                    elif hasattr(block, "thinking"):
                                        reasoning_text = getattr(block, "thinking", None)
                                
                                if block_type in ("reasoning", "thinking") and reasoning_text:
                                    reasoning_blocks.append(reasoning_text)
                            
                            if reasoning_blocks:
                                for i, reasoning_text in enumerate(reasoning_blocks, 1):
                                    logger.info(
                                        f"[ReasoningLogger] 🧠 REASONING #{i} для {self.client_phone}:\n"
                                        f"{reasoning_text[:3000]}"
                                    )
                                    reasoning_content = reasoning_text
                    except Exception as e:
                        logger.debug(f"[ReasoningLogger] Ошибка извлечения content_blocks: {e}")
                
                # Проверяем usage_metadata для reasoning tokens
                if hasattr(message_for_reasoning, "usage_metadata"):
                    try:
                        usage_metadata = message_for_reasoning.usage_metadata
                        if usage_metadata:
                            reasoning_tokens = None
                            if isinstance(usage_metadata, dict):
                                output_details = usage_metadata.get("output_token_details", {})
                                if isinstance(output_details, dict):
                                    reasoning_tokens = output_details.get("reasoning_tokens", 0)
                            elif hasattr(usage_metadata, "output_token_details"):
                                output_details = usage_metadata.output_token_details
                                if hasattr(output_details, "reasoning_tokens"):
                                    reasoning_tokens = output_details.reasoning_tokens
                            
                            if reasoning_tokens and reasoning_tokens > 0:
                                logger.info(
                                    f"[ReasoningLogger] REASONING TOKENS для {self.client_phone}: "
                                    f"{reasoning_tokens} tokens"
                                )
                    except Exception as e:
                        logger.debug(f"[ReasoningLogger] Ошибка извлечения usage_metadata: {e}")
                
                # Также проверяем response_metadata для reasoning tokens и content
                if hasattr(message_for_reasoning, "response_metadata"):
                    response_metadata = message_for_reasoning.response_metadata or {}
                    
                    # Token usage details с reasoning tokens
                    if "token_usage" in response_metadata:
                        token_usage = response_metadata.get("token_usage", {})
                        if isinstance(token_usage, dict):
                            reasoning_tokens = (
                                token_usage.get("reasoning_tokens") or 
                                token_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) or
                                token_usage.get("output_token_details", {}).get("reasoning_tokens", 0)
                            )
                            if reasoning_tokens and reasoning_tokens > 0:
                                logger.info(
                                    f"[ReasoningLogger] REASONING TOKENS (metadata) для {self.client_phone}: "
                                    f"{reasoning_tokens} tokens"
                                )
                    
                    # Прямой reasoning в metadata
                    if "reasoning" in response_metadata and not reasoning_content:
                        reasoning_text = response_metadata.get("reasoning", "")
                        if reasoning_text:
                            logger.info(
                                f"[ReasoningLogger] 🧠 REASONING (metadata) для {self.client_phone}:\n"
                                f"{reasoning_text[:3000]}"
                            )
                            reasoning_content = reasoning_text
            
            # Извлекаем имена инструментов
            tool_names = []
            if tool_calls_info:
                for tc in tool_calls_info:
                    if isinstance(tc, dict):
                        tool_name = tc.get("name") or tc.get("function", {}).get("name", "unknown")
                    elif hasattr(tc, "name"):
                        tool_name = tc.name
                    elif hasattr(tc, "function"):
                        tool_name = getattr(tc.function, "name", "unknown")
                    else:
                        tool_name = "unknown"
                    tool_names.append(tool_name)
            
            # Логируем результат (одно сообщение вместо нескольких)
            if tool_calls_info and tool_names:
                logger.info(
                    f"[ReasoningLogger] LLM END для {self.client_phone}: "
                    f"✅ MODEL DECIDED TO CALL TOOLS: {tool_names}"
                )
            else:
                logger.info(
                    f"[ReasoningLogger] LLM END для {self.client_phone}: "
                    f"❌ MODEL DECIDED NOT TO CALL TOOLS"
                )
            
            # Сохраняем информацию о вызове
            self._llm_calls.append({
                "run_id": run_id,
                "tool_calls": tool_calls_info,
                "tool_names": tool_names,
                "content": content,
                "reasoning_content": reasoning_content,
                "response_metadata": response_metadata,
            })
            
        except Exception as e:
            logger.error(
                f"[ReasoningLogger] Ошибка в on_llm_end: {e}", exc_info=True
            )

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
            logger.info(
                f"[ReasoningLogger] TOOL START: {tool_name} для {self.client_phone}"
            )
            
            self._tool_calls.append({
                "run_id": run_id,
                "tool_name": tool_name,
                "input": input_str,
                "status": "started",
            })
        except Exception as e:
            logger.debug(f"[ReasoningLogger] Ошибка в on_tool_start: {e}")

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Вызывается когда инструмент завершает выполнение."""
        try:
            tool_name = kwargs.get("name", "unknown")
            run_id = kwargs.get("run_id", "unknown")
            logger.info(
                f"[ReasoningLogger] TOOL END: {tool_name} для {self.client_phone} "
                f"(output: {len(output) if output else 0} символов)"
            )
            
            # Обновляем информацию о вызове инструмента
            for tool_call in self._tool_calls:
                if tool_call.get("run_id") == run_id:
                    tool_call["status"] = "completed"
                    tool_call["output"] = output[:500] if output else ""
                    break
        except Exception as e:
            logger.debug(f"[ReasoningLogger] Ошибка в on_tool_end: {e}")

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
        # Не логируем здесь, чтобы избежать дублирования
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Вызывается когда chain завершает выполнение."""
        # Не логируем здесь, чтобы избежать дублирования с ProductAgent.run
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

