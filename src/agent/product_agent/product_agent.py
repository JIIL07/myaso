from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import date
from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from src.agent.middleware import (
    create_model_retry_middleware,
    handle_tool_errors,
    save_product_ids_middleware,
)
from src.agent.product_agent.base_agent import BaseAgent
from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.config.settings import settings
from src.constants import (
    AGENT_RECURSION_LIMIT,
    DEFAULT_TEMPERATURE,
    ERROR_MESSAGE_AGENT_FAILED,
    MAX_AGENT_EXECUTION_TIME,
    MAX_AGENT_ITERATIONS,
    PROMPT_NAME_HUMAN_IN_THE_LOOP,
)
from src.services.ai.agent_logger import get_agent_logger
from src.services.ai.openrouter_client import OpenRouterClient
from src.services.ai.prompt import get_all_system_values, get_prompt
from src.services.callbacks.langfuse_callback import (
    create_langfuse_callback_handler,
    flush_langfuse,
    is_langfuse_enabled,
    observe,
    propagate_attributes,
    update_trace,
)
from src.services.memory.memory_utils import is_memory_initialized
from src.tools.execute_sql import execute_sql_query
from src.tools.generate_sql import generate_sql_from_text
from src.tools.get_client_orders import get_client_orders
from src.tools.get_client_profile import get_client_profile
from src.tools.get_product_by_title import get_product_by_title
from src.tools.get_random_products import get_random_products
from src.tools.get_schema import get_database_schema
from src.tools.send_pricelist import send_pricelist
from src.tools.set_photo_requirement import set_photo_requirement
from src.tools.show_product_photos import show_product_photos
from src.tools.vector_search import vector_search
from src.utils.prompts import get_langfuse_label

logger = logging.getLogger(__name__)


class ProductAgent(BaseAgent):

    DEFAULT_SYSTEM_PROMPT = "Ты — ассистент магазина мясной продукции."

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        memory: Optional[Any] = None,
        tools: Optional[list[Any]] = None,
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
            tools = [
                get_client_profile,
                get_client_orders,
                vector_search,
                get_product_by_title,
                get_random_products,
                get_database_schema,
            ]

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT
        self._agent_cache: dict[str, Any] = {}
        self._cached_prompt_hash: Optional[str] = None

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _get_prompt_hash(self, system_prompt: str) -> str:
        return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        return user_input

    def _create_tools(self) -> list[Any]:
        return self.tools

    # ------------------------------------------------------------------
    # Agent creation & caching
    # ------------------------------------------------------------------

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

        if max_iterations is None:
            max_iterations = MAX_AGENT_ITERATIONS
        max_iterations = max(1, min(10000, max_iterations))

        middleware = [
            create_model_retry_middleware(
                max_retries=2,
                backoff_factor=2.0,
                initial_delay=1.0,
                retry_on=(ConnectionError, TimeoutError, asyncio.TimeoutError),
                on_failure="error",
            ),
            handle_tool_errors,
            save_product_ids_middleware,
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=60.0,
                jitter=True,
                retry_on=(ConnectionError, TimeoutError, asyncio.TimeoutError),
                on_failure="return_message",
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
        current_prompt_hash = self._get_prompt_hash(self.SYSTEM_PROMPT)
        agent_tools = tools or self.tools

        if tools is not None:
            tools_hash = str(sorted([getattr(t, "name", str(t)) for t in agent_tools]))
            cache_key = f"{current_prompt_hash}_{tools_hash}"
        else:
            cache_key = current_prompt_hash

        if cache_key not in self._agent_cache:
            agent = await self._create_agent(tools=agent_tools)
            self._agent_cache[cache_key] = agent
            self._cached_prompt_hash = current_prompt_hash
        elif current_prompt_hash != self._cached_prompt_hash:
            self._agent_cache.clear()
            agent = await self._create_agent(tools=agent_tools)
            self._agent_cache[cache_key] = agent
            self._cached_prompt_hash = current_prompt_hash
        else:
            agent = self._agent_cache[cache_key]

        return agent

    # ------------------------------------------------------------------
    # Prompt & context loading
    # ------------------------------------------------------------------

    async def _load_prompt_and_context(
        self,
        prompt_name: Optional[str],
        client_phone: str,
    ) -> tuple[str, dict[str, str], str, list[BaseMessage]]:
        async def load_langfuse_prompt() -> Optional[str]:
            if not prompt_name:
                return None
            try:
                langfuse_label = get_langfuse_label()
                return await get_prompt(
                    prompt_name=prompt_name,
                    default_prompt=self.DEFAULT_SYSTEM_PROMPT,
                    langfuse_label=langfuse_label,
                )
            except Exception as e:
                logger.error("[ProductAgent] Failed to load prompt '%s': %s", prompt_name, e)
                return None

        async def load_system_vars() -> dict[str, str]:
            try:
                return await get_all_system_values()
            except Exception as e:
                logger.error("[ProductAgent] Failed to load system vars: %s", e)
                return {}

        async def load_memory() -> list[BaseMessage]:
            if not self.memory or not is_memory_initialized(self.memory):
                return []
            try:
                memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
                return memory_vars.get("history", [])
            except Exception as e:
                logger.error("[ProductAgent] Memory load error: %s", e, exc_info=True)
                return []

        langfuse_prompt, system_vars, chat_history = await asyncio.gather(
            load_langfuse_prompt(),
            load_system_vars(),
            load_memory(),
        )

        if langfuse_prompt:
            base_prompt = f"{langfuse_prompt}\n\n{self.DEFAULT_SYSTEM_PROMPT}".strip()
        else:
            base_prompt = self.DEFAULT_SYSTEM_PROMPT

        client_info = f"Номер телефона: {client_phone}"
        return base_prompt, system_vars, client_info, chat_history

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _postprocess_response(response: str) -> str:
        if not response:
            return response
        response = re.sub(r"\n{3,}", "\n\n", response)
        response = re.sub(r" {2,}", " ", response)
        return response.strip()

    def _extract_response(self, result: dict[str, Any]) -> Optional[str]:
        messages_result = result.get("messages", [])
        response_text = ""

        for msg in reversed(messages_result):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    response_text = " ".join(text_parts) or str(content)
                else:
                    response_text = str(content) or ""
                break

        if not response_text:
            response_text = result.get("output", "")

        response_text = self._postprocess_response(response_text)

        if not response_text or len(response_text.strip()) < 3:
            return None
        return response_text

    async def _get_hitl_prompt(self) -> Optional[str]:
        try:
            return await get_prompt(
                prompt_name=PROMPT_NAME_HUMAN_IN_THE_LOOP,
                default_prompt=(
                    "Извините, я не смог обработать ваш запрос. "
                    "Пожалуйста, свяжитесь с нашим менеджером для получения помощи."
                ),
                langfuse_label=get_langfuse_label(),
            )
        except Exception as e:
            logger.error("[ProductAgent] HITL prompt load error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Memory persistence
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

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
            if MAX_AGENT_EXECUTION_TIME > 0:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": messages}, config=config, context=context),
                    timeout=MAX_AGENT_EXECUTION_TIME,
                )
            else:
                result = await agent.ainvoke(
                    {"messages": messages}, config=config, context=context
                )
            return result
        except asyncio.TimeoutError as e:
            from src.utils.errors import AgentTimeoutError

            msg = "Agent exceeded max execution time (%s sec)" % MAX_AGENT_EXECUTION_TIME
            logger.error("[ProductAgent] Timeout: %s", msg)
            raise AgentTimeoutError(msg, {"timeout": MAX_AGENT_EXECUTION_TIME}) from e
        except Exception as e:
            from src.utils.errors import AgentExecutionError

            msg = "Agent execution error: %s" % e
            logger.error("[ProductAgent] Error: %s", msg, exc_info=True)
            raise AgentExecutionError(msg, {"original_error": str(e)}) from e

    # ------------------------------------------------------------------
    # Main entry point  (Langfuse v3: @observe)
    # ------------------------------------------------------------------

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
                self.llm.temperature = DEFAULT_TEMPERATURE

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
        # --- Prompt & history -----------------------------------------------
        chat_history: list[BaseMessage] = []

        if system_prompt:
            final_prompt = system_prompt
            if self.memory and is_memory_initialized(self.memory):
                try:
                    memory_vars = await self.memory.load_memory_variables(
                        {}, return_messages=True
                    )
                    chat_history = memory_vars.get("history", [])
                except Exception as e:
                    logger.error("[ProductAgent] Memory load error: %s", e, exc_info=True)
        else:
            base_prompt, _sys_vars, _client_info, chat_history = (
                await self._load_prompt_and_context(
                    prompt_name=None,
                    client_phone=client_phone,
                )
            )
            final_prompt = base_prompt

        self.SYSTEM_PROMPT = final_prompt

        # --- Tools ----------------------------------------------------------
        context_tools = [set_photo_requirement]
        sql_tools = [generate_sql_from_text, execute_sql_query, get_database_schema]
        media_tools = [show_product_photos, send_pricelist]
        agent_tools = self.tools + sql_tools + media_tools + context_tools

        # --- Callbacks (Langfuse v3 + agent logger) -------------------------
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
            "recursion_limit": AGENT_RECURSION_LIMIT,
        }

        # --- Update Langfuse trace with input --------------------------------
        update_trace(
            name=trace_name,
            user_id=client_phone,
            session_id=f"{client_phone}_{date.today()}",
            input={"query": user_input},
            tags=["product_agent"],
        )

        # --- Build messages --------------------------------------------------
        messages: list[BaseMessage] = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=user_input))

        # --- Execute ---------------------------------------------------------
        try:
            result = await self._execute_agent(
                messages, agent_tools, config, client_phone
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.error("[ProductAgent] Agent error: %s", e, exc_info=True)
            hitl_prompt = await self._get_hitl_prompt()
            if hitl_prompt:
                return hitl_prompt
            raise

        # --- Extract response ------------------------------------------------
        response_text = self._extract_response(result)

        if not response_text or not response_text.strip():
            logger.warning("[ProductAgent] Empty agent response, using HITL")
            hitl_prompt = await self._get_hitl_prompt()
            if hitl_prompt:
                return hitl_prompt
            response_text = (
                "Извините, я не смог обработать ваш запрос. "
                "Пожалуйста, попробуйте переформулировать вопрос."
            )

        # --- Update trace with output ----------------------------------------
        update_trace(output={"response": response_text[:500]})

        # --- Persist to memory -----------------------------------------------
        await self._save_to_memory(user_input, response_text, client_phone)

        # --- Flush Langfuse --------------------------------------------------
        flush_langfuse()

        return response_text
