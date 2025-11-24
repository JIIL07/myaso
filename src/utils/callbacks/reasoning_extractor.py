"""Утилита для извлечения reasoning и thinking из LLM ответов."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage


class ReasoningExtractor:
    """Класс для извлечения reasoning и thinking из LLM ответов."""

    @staticmethod
    def extract_reasoning_from_response(response: Any) -> Dict[str, Any]:
        """Извлекает reasoning и thinking из LLM ответа.

        Args:
            response: Ответ от LLM (может быть в разных форматах)

        Returns:
            Словарь с извлеченными данными:
            {
                "reasoning_text": Optional[str],
                "thinking_text": Optional[str],
                "reasoning_tokens": Optional[int],
                "has_reasoning": bool
            }
        """
        result = {
            "reasoning_text": None,
            "thinking_text": None,
            "reasoning_tokens": None,
            "has_reasoning": False,
        }

        # Находим AIMessage в ответе
        message = ReasoningExtractor._find_message_in_response(response)
        if not message:
            return result

        # Извлекаем из content_blocks
        reasoning_text, thinking_text = ReasoningExtractor._extract_from_content_blocks(message)
        if reasoning_text:
            result["reasoning_text"] = reasoning_text
            result["has_reasoning"] = True
        if thinking_text:
            result["thinking_text"] = thinking_text
            result["has_reasoning"] = True

        # Извлекаем reasoning tokens из usage_metadata
        if hasattr(message, "usage_metadata"):
            reasoning_tokens = ReasoningExtractor._extract_from_usage_metadata(message.usage_metadata)
            if reasoning_tokens:
                result["reasoning_tokens"] = reasoning_tokens
                if reasoning_tokens > 0:
                    result["has_reasoning"] = True

        # Извлекаем reasoning tokens из response_metadata.token_usage
        if hasattr(message, "response_metadata"):
            response_metadata = message.response_metadata or {}
            if isinstance(response_metadata, dict) and "token_usage" in response_metadata:
                token_usage = response_metadata.get("token_usage")
                if token_usage and isinstance(token_usage, dict):
                    # Проверяем reasoning_tokens в разных местах token_usage
                    reasoning_tokens_from_metadata = (
                        token_usage.get("reasoning_tokens") or
                        (token_usage.get("completion_tokens_details", {}).get("reasoning_tokens") if isinstance(token_usage.get("completion_tokens_details"), dict) else None) or
                        (token_usage.get("output_token_details", {}).get("reasoning_tokens") if isinstance(token_usage.get("output_token_details"), dict) else None)
                    )
                    if reasoning_tokens_from_metadata and not result["reasoning_tokens"]:
                        result["reasoning_tokens"] = reasoning_tokens_from_metadata
                        if reasoning_tokens_from_metadata > 0:
                            result["has_reasoning"] = True

            # Извлекаем reasoning текст из response_metadata
            metadata_reasoning = ReasoningExtractor._extract_from_response_metadata(response_metadata)
            if metadata_reasoning and not result["reasoning_text"]:
                result["reasoning_text"] = metadata_reasoning
                result["has_reasoning"] = True

        return result

    @staticmethod
    def _find_message_in_response(response: Any) -> Optional[AIMessage]:
        """Находит AIMessage в разных форматах ответа.

        Args:
            response: Ответ от LLM

        Returns:
            AIMessage если найден, иначе None
        """
        # Формат 1: response.generations[0][0].message (стандартный LangChain)
        if hasattr(response, "generations") and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "message"):
                        message = generation.message
                        if isinstance(message, AIMessage):
                            return message

        # Формат 2: response.message (прямой доступ)
        if hasattr(response, "message"):
            message = response.message
            if isinstance(message, AIMessage):
                return message

        return None

    @staticmethod
    def _extract_from_content_blocks(message: AIMessage) -> tuple[Optional[str], Optional[str]]:
        """Извлекает reasoning и thinking из content_blocks.

        Args:
            message: AIMessage с content_blocks

        Returns:
            Кортеж (reasoning_text, thinking_text)
        """
        reasoning_text = None
        thinking_text = None

        if not hasattr(message, "content_blocks"):
            return None, None

        try:
            content_blocks = message.content_blocks
            if not content_blocks:
                return None, None

            for block in content_blocks:
                block_type = None
                block_reasoning = None
                block_thinking = None

                # Проверяем разные форматы блоков
                if isinstance(block, dict):
                    block_type = block.get("type")
                    block_reasoning = block.get("reasoning")
                    block_thinking = block.get("thinking")
                elif hasattr(block, "type"):
                    block_type = getattr(block, "type", None)
                    if hasattr(block, "reasoning"):
                        block_reasoning = getattr(block, "reasoning", None)
                    if hasattr(block, "thinking"):
                        block_thinking = getattr(block, "thinking", None)

                # Извлекаем reasoning
                if block_type == "reasoning" and block_reasoning:
                    if reasoning_text:
                        reasoning_text += "\n\n" + str(block_reasoning)
                    else:
                        reasoning_text = str(block_reasoning)

                # Извлекаем thinking
                if block_type == "thinking" and block_thinking:
                    if thinking_text:
                        thinking_text += "\n\n" + str(block_thinking)
                    else:
                        thinking_text = str(block_thinking)

        except Exception:
            pass  # Не логируем ошибки извлечения

        return reasoning_text, thinking_text

    @staticmethod
    def _extract_from_usage_metadata(usage_metadata: Any) -> Optional[int]:
        """Извлекает reasoning_tokens из usage_metadata.

        Args:
            usage_metadata: Метаданные использования токенов

        Returns:
            Количество reasoning tokens или None
        """
        if not usage_metadata:
            return None

        try:
            if isinstance(usage_metadata, dict):
                output_details = usage_metadata.get("output_token_details")
                if output_details and isinstance(output_details, dict):
                    return output_details.get("reasoning_tokens")
            elif hasattr(usage_metadata, "output_token_details"):
                output_details = usage_metadata.output_token_details
                if output_details and hasattr(output_details, "reasoning_tokens"):
                    return output_details.reasoning_tokens
        except Exception:
            pass

        return None

    @staticmethod
    def _extract_from_response_metadata(metadata: Dict[str, Any]) -> Optional[str]:
        """Извлекает reasoning из response_metadata.

        Args:
            metadata: response_metadata из сообщения

        Returns:
            Текст reasoning или None
        """
        if not metadata or not isinstance(metadata, dict):
            return None

        try:
            # Прямое поле reasoning
            if "reasoning" in metadata:
                reasoning = metadata.get("reasoning")
                if reasoning:
                    return str(reasoning)

            # Поле thinking
            if "thinking" in metadata:
                thinking = metadata.get("thinking")
                if thinking:
                    return str(thinking)

        except Exception:
            pass

        return None

