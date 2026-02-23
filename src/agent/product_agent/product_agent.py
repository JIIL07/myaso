from __future__ import annotations

import asyncio
import logging
import re
from datetime import date
from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ToolRetryMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.middleware import (
    create_model_retry_middleware,
    handle_tool_errors,
    save_product_ids_middleware,
)
from src.agent.product_agent.base_agent import BaseAgent
from src.agent.product_agent.policy import AgentPolicy, get_agent_policy
from src.agent.product_agent.tool_registry import (
    ToolRegistryFlags,
    build_agent_tools,
    get_core_tools,
)
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.constants import (
    ERROR_MESSAGE_AGENT_FAILED,
    ERROR_MESSAGE_HITL_FALLBACK,
    PROMPT_NAME_HUMAN_IN_THE_LOOP,
    PROMPT_NAME_SYSTEM_PROMPT,
)
from src.services.ai.agent_logger import get_agent_logger
from src.services.ai.openrouter_client import OpenRouterClient
from src.services.ai.prompt import get_prompt
from src.services.callbacks.langfuse_callback import (
    create_langfuse_callback_handler,
    flush_langfuse,
    observe,
    propagate_attributes,
    update_trace,
)
from src.services.memory.memory_utils import is_memory_initialized
from src.utils.prompts import get_langfuse_label

logger = logging.getLogger(__name__)


class ProductAgent(BaseAgent):

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        memory: Optional[Any] = None,
        tools: Optional[list[Any]] = None,
        policy: Optional[AgentPolicy] = None,
        tool_flags: Optional[ToolRegistryFlags] = None,
        **kwargs: Any,
    ) -> None:
        if llm is None:
            try:
                openrouter_client = OpenRouterClient()
                llm = openrouter_client.get_llm()
            except Exception as e:
                logger.error("[ProductAgent] LLM init error: %s", e, exc_info=True)
                raise ValueError("Failed to initialize LLM: %s" % e) from e

        if tools is None:
            tools = get_core_tools()

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.policy = policy or get_agent_policy()
        self.tool_flags = tool_flags or ToolRegistryFlags()
        self.SYSTEM_PROMPT = ""

    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        return user_input

    def _create_tools(self) -> list[Any]:
        return self.tools

    async def _create_agent(
        self,
        tools: Optional[list[Any]] = None,
        max_iterations: Optional[int] = None,
    ) -> Any:
        if not self.llm:
            raise ValueError("LLM not initialized")

        system_prompt = self.SYSTEM_PROMPT or ""
        agent_tools = tools or self.tools

        if not agent_tools:
            logger.warning("[ProductAgent] Tool list is empty")

        max_iterations = self.policy.clamp_iterations(max_iterations)
        retry_on = (ConnectionError, TimeoutError, asyncio.TimeoutError)

        middleware = [
            create_model_retry_middleware(
                max_retries=self.policy.model_retry.max_retries,
                backoff_factor=self.policy.model_retry.backoff_factor,
                initial_delay=self.policy.model_retry.initial_delay,
                max_delay=self.policy.model_retry.max_delay,
                jitter=self.policy.model_retry.jitter,
                retry_on=retry_on,
                on_failure=self.policy.model_retry.on_failure,
            ),
            handle_tool_errors,
            save_product_ids_middleware,
            ToolRetryMiddleware(
                max_retries=self.policy.tool_retry.max_retries,
                backoff_factor=self.policy.tool_retry.backoff_factor,
                initial_delay=self.policy.tool_retry.initial_delay,
                max_delay=self.policy.tool_retry.max_delay,
                jitter=self.policy.tool_retry.jitter,
                retry_on=retry_on,
                on_failure=self.policy.tool_retry.on_failure,
            ),
        ]
        if max_iterations > 0:
            middleware.append(
                ModelCallLimitMiddleware(run_limit=max_iterations, exit_behavior="end")
            )

        return create_agent(
            model=self.llm,
            tools=agent_tools,
            system_prompt=system_prompt,
            middleware=middleware or None,
            state_schema=ProductAgentState,
            context_schema=ProductAgentContext,
        )

    async def _get_agent(self, tools: Optional[list[Any]] = None) -> Any:
        agent_tools = tools or self.tools
        return await self._create_agent(tools=agent_tools)

    async def _load_chat_history(self) -> list[BaseMessage]:
        if not self.memory or not is_memory_initialized(self.memory):
            return []
        try:
            memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
            return memory_vars.get("history", [])
        except Exception as e:
            logger.error("[ProductAgent] Memory load error: %s", e, exc_info=True)
            return []

    async def _resolve_system_prompt(self, system_prompt: Optional[str]) -> str:
        if system_prompt and system_prompt.strip():
            return system_prompt
        prompt = await get_prompt(
            prompt_name=PROMPT_NAME_SYSTEM_PROMPT,
            default_prompt=None,
            langfuse_label=get_langfuse_label(),
            required=True,
        )
        return prompt or ""

    @staticmethod
    def _postprocess_response(response: str) -> str:
        if not response:
            return response
        response = re.sub(r"\n{3,}", "\n\n", response)
        response = re.sub(r" {2,}", " ", response)
        return response.strip()

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts) or str(content)
        return str(content) or ""

    def _extract_response(self, result: dict[str, Any]) -> Optional[str]:
        response_text = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                response_text = self._message_content_to_text(msg.content)
                break

        response_text = response_text or result.get("output", "")
        response_text = self._postprocess_response(response_text)
        return response_text if response_text and len(response_text.strip()) >= 3 else None

    async def _hitl_or_fallback(self) -> str:
        hitl_prompt = await self._get_hitl_prompt()
        return hitl_prompt or ERROR_MESSAGE_HITL_FALLBACK

    async def _get_hitl_prompt(self) -> Optional[str]:
        try:
            return await get_prompt(
                prompt_name=PROMPT_NAME_HUMAN_IN_THE_LOOP,
                default_prompt=None,
                langfuse_label=get_langfuse_label(),
                required=False,
            )
        except Exception as e:
            logger.error("[ProductAgent] HITL prompt load error: %s", e)
            return None

    async def _save_to_memory(
        self,
        user_input: str,
        response_text: str,
        client_phone: str,
    ) -> None:
        if self.memory is None:
            return
        try:
            if not is_memory_initialized(self.memory):
                logger.warning(
                    "[ProductAgent] Memory not initialized for %s, skipping save",
                    client_phone,
                )
                return
            await self.memory.add_messages([
                HumanMessage(content=user_input),
                AIMessage(content=response_text),
            ])
        except Exception as e:
            logger.error(
                "[ProductAgent] Failed to save memory for %s: %s",
                client_phone, e, exc_info=True,
            )

    async def _execute_agent(
        self,
        messages: list[BaseMessage],
        agent_tools: list[Any],
        config: RunnableConfig,
        client_phone: str,
    ) -> dict[str, Any]:
        agent = await self._get_agent(tools=agent_tools)
        context = ProductAgentContext(client_phone=client_phone)

        try:
            if self.policy.execution_timeout_seconds > 0:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": messages}, config=config, context=context),
                    timeout=self.policy.execution_timeout_seconds,
                )
            else:
                result = await agent.ainvoke(
                    {"messages": messages}, config=config, context=context
                )
            return result
        except asyncio.TimeoutError as e:
            from src.utils.errors import AgentTimeoutError

            msg = "Agent exceeded max execution time (%s sec)" % self.policy.execution_timeout_seconds
            logger.error("[ProductAgent] Timeout: %s", msg)
            raise AgentTimeoutError(msg, {"timeout": self.policy.execution_timeout_seconds}) from e
        except Exception as e:
            from src.utils.errors import AgentExecutionError

            msg = "Agent execution error: %s" % e
            logger.error("[ProductAgent] Error: %s", msg, exc_info=True)
            raise AgentExecutionError(msg, {"original_error": str(e)}) from e

    @observe(name="product-agent-run")
    async def run(
        self,
        user_input: str,
        client_phone: str,
        endpoint_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        trace_name = endpoint_name or "ProductAgent"
        session_id = f"{client_phone}_{date.today()}"

        try:
            if hasattr(self.llm, "temperature"):
                self.llm.temperature = self.policy.default_temperature

            with propagate_attributes(session_id=session_id, user_id=client_phone):
                return await self._run_core(
                    user_input=user_input,
                    client_phone=client_phone,
                    trace_name=trace_name,
                    system_prompt=system_prompt,
                )
        except Exception as e:
            logger.error("[ProductAgent] Error: %s", e, exc_info=True)
            flush_langfuse()
            return ERROR_MESSAGE_AGENT_FAILED

    async def _run_core(
        self,
        *,
        user_input: str,
        client_phone: str,
        trace_name: str,
        system_prompt: Optional[str],
    ) -> str:
        chat_history = await self._load_chat_history()
        final_prompt = await self._resolve_system_prompt(system_prompt)

        self.SYSTEM_PROMPT = final_prompt

        agent_tools = build_agent_tools(self.tools, self.tool_flags)

        agent_logger = get_agent_logger()
        agent_logger_callback = agent_logger.get_callback_handler(client_phone)

        callbacks_list: list[Any] = [agent_logger_callback]
        langfuse_cb = create_langfuse_callback_handler()
        if langfuse_cb is not None:
            callbacks_list.insert(0, langfuse_cb)

        config: RunnableConfig = {
            "callbacks": callbacks_list,
            "metadata": {
                "phone": client_phone,
                "user_id": client_phone,
                "trace_name": trace_name,
            },
            "run_name": trace_name,
            "tags": ["product_agent", "conversation", trace_name],
            "recursion_limit": self.policy.recursion_limit,
        }

        update_trace(
            name=trace_name,
            user_id=client_phone,
            session_id=f"{client_phone}_{date.today()}",
            input={"query": user_input},
            tags=["product_agent"],
        )

        messages: list[BaseMessage] = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=user_input))

        try:
            result = await self._execute_agent(
                messages, agent_tools, config, client_phone
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.error("[ProductAgent] Agent error: %s", e, exc_info=True)
            return await self._hitl_or_fallback()

        response_text = self._extract_response(result)

        if not response_text or not response_text.strip():
            logger.warning("[ProductAgent] Empty agent response, using HITL")
            response_text = await self._hitl_or_fallback()

        update_trace(output={"response": response_text[:500]})
        await self._save_to_memory(user_input, response_text, client_phone)
        flush_langfuse()

        return response_text
