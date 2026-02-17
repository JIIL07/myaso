"""ProductAgent — агент для работы с продуктами и каталогом с помощью LangChain tools."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from src.services.ai.openrouter_client import OpenRouterClient

from src.agent.product_agent.types import ProductAgentContext, ProductAgentState
from src.tools.product_tools import (
    get_database_schema,
    get_product_by_title,
    get_random_products,
    execute_sql_query,
    generate_sql_from_text,
    get_products_table_schema as get_table_schema,
)
from src.tools.client_tools import get_client_orders, get_client_profile
from src.tools.media_tools import send_pricelist, show_product_photos
from src.tools.vector_tools import vector_search

from src.agent.product_agent.base_agent import BaseAgent
from src.agent.middleware import (
    create_model_retry_middleware,
    save_product_ids_middleware,
    handle_tool_errors,
)
from src.config.settings import settings
from src.services.ai.constants import (
    AGENT_RECURSION_LIMIT,
    DEFAULT_TEMPERATURE,
    MAX_AGENT_EXECUTION_TIME,
    MAX_AGENT_ITERATIONS,
)
from src.services.ai.prompt import (
    get_all_system_values,
    get_prompt,
)
from src.utils.prompts import get_langfuse_label
from src.services.callbacks.langfuse_callback import LangfuseHandler
from src.services.memory.memory_utils import is_memory_initialized
from src.services.ai.agent_logger import get_agent_logger

try:
    from langfuse import get_client, propagate_attributes
    LANGFUSE_SDK_AVAILABLE = True
except ImportError:
    LANGFUSE_SDK_AVAILABLE = False
    get_client = None
    propagate_attributes = None

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class ProductAgent(BaseAgent):
    """Агент для обработки запросов пользователей о товарах и каталоге."""

    DEFAULT_SYSTEM_PROMPT = "Ты — ассистент магазина мясной продукции."

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        memory: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Инициализация ProductAgent.

        Args:
            llm: LangChain LLM модель (если None, создаётся ChatOpenAI)
            retriever: Векторный ретривер (опционально, для будущего использования)
            memory: Память диалога (BaseChatMessageHistory)
            tools: Список инструментов (если None, используются стандартные)
            **kwargs: Дополнительные параметры для BaseAgent

        Note:
            Агент создается через LangChain create_agent API, который автоматически
            определяет тип агента на основе модели и инструментов.
        """
        if llm is None:
            try:
                openrouter_client = OpenRouterClient()
                llm = openrouter_client.get_llm()
            except Exception as e:
                logger.error(
                    f"[ProductAgent] Ошибка инициализации LLM: {e}",
                    exc_info=True
                )
                raise ValueError(f"Не удалось инициализировать LLM: {e}") from e

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

    def _get_prompt_hash(self, system_prompt: str) -> str:
        """Возвращает хеш системного промпта для кэширования агента."""
        return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()

    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        """Совместимость с BaseAgent: возвращает user_input без изменений."""
        return user_input

    def _create_tools(self) -> List[Any]:
        """Совместимость с BaseAgent: возвращает текущий список tools."""
        return self.tools

    async def _create_agent(
        self, tools: Optional[List[Any]] = None, max_iterations: Optional[int] = None
    ) -> Any:
        """Создаёт агента через create_agent API с валидацией конфигурации.

        Примечание: Использование ToolNode было рассмотрено, но не реализовано,
        так как текущая реализация с middleware (handle_tool_errors, ToolRetryMiddleware)
        обеспечивает достаточный контроль над обработкой ошибок инструментов.
        ToolNode может быть полезен в будущем для более гранулярного контроля,
        но требует рефакторинга существующей логики middleware.

        Args:
            tools: Список инструментов (если None, используются self.tools)
            max_iterations: Максимальное количество итераций (загружается из БД если None)

        Returns:
            Runnable объект агента
            
        Raises:
            ValueError: Если конфигурация невалидна
        """
        if not self.llm:
            raise ValueError("LLM не инициализирован")
        
        system_prompt = self.SYSTEM_PROMPT or ""
        agent_tools = tools or self.tools
        
        if not agent_tools:
            logger.warning("[ProductAgent._create_agent] Список инструментов пуст")

        if max_iterations is None:
            max_iterations = MAX_AGENT_ITERATIONS
        
        max_iterations = max(1, min(10000, max_iterations))

        tool_retry_max_retries = 3
        tool_retry_backoff_factor = 2.0
        tool_retry_initial_delay = 1.0

        model_retry_max_retries = 2
        model_retry_backoff_factor = 2.0
        model_retry_initial_delay = 1.0

        middleware = [
            create_model_retry_middleware(
                max_retries=model_retry_max_retries,
                backoff_factor=model_retry_backoff_factor,
                initial_delay=model_retry_initial_delay,
                retry_on=(ConnectionError, TimeoutError, asyncio.TimeoutError),
                on_failure="error",
            ),
            handle_tool_errors,
            save_product_ids_middleware,
            ToolRetryMiddleware(
                max_retries=tool_retry_max_retries,
                backoff_factor=tool_retry_backoff_factor,
                initial_delay=tool_retry_initial_delay,
                max_delay=60.0,
                jitter=True,
                retry_on=(ConnectionError, TimeoutError, asyncio.TimeoutError),
                on_failure="return_message",
            ),
        ]
        if max_iterations > 0:
            middleware.append(
                ModelCallLimitMiddleware(
                    run_limit=max_iterations,
                    exit_behavior="end",
                )
            )

        agent = create_agent(
            model=self.llm,
            tools=agent_tools,
            system_prompt=system_prompt,
            middleware=middleware if middleware else None,
            state_schema=ProductAgentState,
            context_schema=ProductAgentContext,
        )

        return agent

    async def _get_agent(
        self, tools: Optional[List[Any]] = None
    ) -> Any:
        """Получает агента из кэша или создает новый.

        Кэширует агента по хешу текущего SYSTEM_PROMPT и инструментов.
        Если промпт или инструменты изменились, создает новый агент.

        Args:
            tools: Список инструментов (если None, используются self.tools)

        Returns:
            Runnable объект агента
        """
        current_prompt_hash = self._get_prompt_hash(self.SYSTEM_PROMPT)
        agent_tools = tools or self.tools

        if tools is not None:
            tools_hash = str(sorted([getattr(t, 'name', str(t)) for t in agent_tools]))
            cache_key = f"{current_prompt_hash}_{tools_hash}"
        else:
            cache_key = current_prompt_hash

        if cache_key not in self._agent_cache:
            agent = await self._create_agent(tools=agent_tools)
            self._agent_cache[cache_key] = agent
            self._cached_prompt_hash = current_prompt_hash
        else:
            if current_prompt_hash != self._cached_prompt_hash:
                self._agent_cache.clear()
                agent = await self._create_agent(tools=agent_tools)
                self._agent_cache[cache_key] = agent
                self._cached_prompt_hash = current_prompt_hash
            else:
                agent = self._agent_cache[cache_key]

        return agent

    async def _load_prompt_and_context(
        self,
        prompt_name: Optional[str],
        client_phone: str,
    ) -> tuple[str, Dict[str, str], str, List[BaseMessage]]:
        """Загружает промпт, системные переменные, информацию о клиенте и историю.

        Загружает данные параллельно для улучшения производительности.

        Args:
            prompt_name: Название промпта в Langfuse
            client_phone: Номер телефона клиента

        Returns:
            Кортеж (base_prompt, system_vars, client_info, chat_history)
        """
        async def load_langfuse_prompt() -> Optional[str]:
            """Загружает промпт из Langfuse."""
            if not prompt_name:
                return None
            try:
                langfuse_label = get_langfuse_label()
                langfuse_prompt = await get_prompt(
                    prompt_name=prompt_name,
                    default_prompt=self.DEFAULT_SYSTEM_PROMPT,
                    langfuse_label=langfuse_label,
                )
                if not langfuse_prompt:
                    logger.warning(f"[ProductAgent] Промпт '{prompt_name}' не найден в Langfuse, используется дефолтный")
                return langfuse_prompt
            except Exception as e:
                logger.error(f"[ProductAgent] Не удалось загрузить промпт '{prompt_name}': {e}")
                return None

        async def load_system_vars() -> Dict[str, str]:
            """Загружает системные переменные."""
            try:
                return await get_all_system_values()
            except Exception as e:
                logger.error(f"[ProductAgent] Не удалось загрузить системные переменные: {e}")
                return {}

        async def load_memory() -> List[BaseMessage]:
            """Загружает историю диалога."""
            if not self.memory or not is_memory_initialized(self.memory):
                return []
            try:
                memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
                return memory_vars.get("history", [])
            except Exception as e:
                logger.error(f"[ProductAgent] Ошибка загрузки памяти: {e}", exc_info=True)
                return []

        # Параллельная загрузка всех данных
        langfuse_prompt, system_vars, chat_history = await asyncio.gather(
            load_langfuse_prompt(),
            load_system_vars(),
            load_memory(),
        )

        # Сборка финального промпта
        if langfuse_prompt:
            base_prompt = f"{langfuse_prompt}\n\n{self.DEFAULT_SYSTEM_PROMPT}".strip()
        else:
            base_prompt = self.DEFAULT_SYSTEM_PROMPT

        client_info = f"Номер телефона: {client_phone}"

        return base_prompt, system_vars, client_info, chat_history

    def _prepare_messages(
        self,
        user_input: str,
        chat_history: List[BaseMessage],
    ) -> str:
        """Подготавливает сообщения для агента.

        Args:
            user_input: Текст запроса пользователя
            chat_history: История сообщений (передается агенту напрямую)

        Returns:
            Текст запроса пользователя без изменений
        """
        return user_input

    async def _execute_agent(
        self,
        messages: List[BaseMessage],
        agent_tools: List[Any],
        config: RunnableConfig,
        client_phone: str,
    ) -> Dict[str, Any]:
        """Выполняет агента с заданными сообщениями и инструментами с timeout и валидацией.

        Примечание: 
        - Retry для вызовов модели обрабатывается через ModelRetryMiddleware
        - Retry для инструментов обрабатывается через ToolRetryMiddleware
        - Этот метод обрабатывает только timeout и финальные ошибки выполнения агента

        Args:
            messages: Список сообщений для агента
            agent_tools: Список инструментов агента
            config: Конфигурация для выполнения
            client_phone: Номер телефона клиента для передачи в context

        Returns:
            Результат выполнения агента

        Raises:
            ValueError: Если входные данные невалидны
            AgentTimeoutError: Если агент превысил время выполнения
            AgentExecutionError: Если произошла ошибка при выполнении агента после всех попыток
        """
        agent = await self._get_agent(tools=agent_tools)
        
        context = ProductAgentContext(client_phone=client_phone)
        
        max_execution_time = MAX_AGENT_EXECUTION_TIME
        
        try:
            if max_execution_time > 0:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": messages}, config=config, context=context),
                    timeout=max_execution_time,
                )
            else:
                result = await agent.ainvoke({"messages": messages}, config=config, context=context)
            return result
        except asyncio.TimeoutError as e:
            from src.utils.errors.exceptions import AgentTimeoutError
            error_msg = f"Агент превысил максимальное время выполнения ({max_execution_time} секунд)"
            logger.error(f"[ProductAgent._execute_agent] Timeout агента: {error_msg}")
            raise AgentTimeoutError(error_msg, {"timeout": max_execution_time}) from e
        except Exception as e:
            from src.utils.errors.exceptions import AgentExecutionError
            error_msg = f"Ошибка при выполнении агента: {str(e)}"
            logger.error(f"[ProductAgent._execute_agent] Ошибка агента: {error_msg}", exc_info=True)
            raise AgentExecutionError(error_msg, {"original_error": str(e)}) from e

    def _postprocess_response(self, response: str) -> str:
        """Постобработка ответа агента.
        
        - Удаляет артефакты инструментов
        - Очищает лишние пробелы
        - Удаляет служебные метки
        
        Args:
            response: Исходный ответ агента
        
        Returns:
            Обработанный ответ
        """
        if not response:
            return response
        
        import re
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        return response.strip()

    async def _get_hitl_prompt(self) -> Optional[str]:
        """Получает промпт для вызова человека (HITL).

        Returns:
            Текст промпта HITL или None, если не найден
        """
        from src.services.langfuse.prompt_names import PROMPT_NAME_HUMAN_IN_THE_LOOP
        from src.services.ai.prompt import get_prompt
        from src.utils.prompts import get_langfuse_label
        
        try:
            hitl_prompt = await get_prompt(
                prompt_name=PROMPT_NAME_HUMAN_IN_THE_LOOP,
                default_prompt="Извините, я не смог обработать ваш запрос. Пожалуйста, свяжитесь с нашим менеджером для получения помощи.",
                langfuse_label=get_langfuse_label(),
            )
            return hitl_prompt
        except Exception as e:
            logger.error(f"[ProductAgent._get_hitl_prompt] Ошибка загрузки HITL промпта: {e}")
            return None

    def _extract_response(self, result: Dict[str, Any]) -> Optional[str]:
        """Извлекает ответ агента из результата выполнения.

        Args:
            result: Результат выполнения агента

        Returns:
            Текст ответа агента или None, если ответ пустой (будет использован HITL промпт)
        """
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

    async def _save_to_memory(
        self,
        user_input: str,
        response_text: str,
        chat_history: List[BaseMessage],
        client_phone: str,
    ) -> None:
        """Сохраняет сообщения в память.

        Args:
            user_input: Текст запроса пользователя
            response_text: Текст ответа агента
            chat_history: История сообщений для определения первого сообщения
            client_phone: Номер телефона клиента
        """
        if self.memory is None:
            return

        try:
            if not is_memory_initialized(self.memory):
                logger.warning(f"[ProductAgent] Память не инициализирована для {client_phone}, пропускаем сохранение")
                return

            await self.memory.add_messages([
                HumanMessage(content=user_input),
                AIMessage(content=response_text),
            ])
        except Exception as e:
            logger.error(f"[ProductAgent] Не удалось сохранить в память для {client_phone}: {e}", exc_info=True)

    async def run(
        self,
        user_input: str,
        client_phone: str,
        endpoint_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Запускает агента для обработки запроса пользователя.

        Использует современный LangChain API create_agent для выполнения агента.
        Формат вызова: {"messages": [...]} вместо старого {"input": ..., "chat_history": ...}.
        Результат: {"messages": [...]} с последним AIMessage в качестве ответа.

        Args:
            user_input: Текст запроса пользователя
            client_phone: Номер телефона клиента
            endpoint_name: Имя endpoint для трейсинга
            system_prompt: Готовый системный промпт (если передан, используется вместо загрузки)

        Returns:
            Строка с ответом агента (извлекается из последнего AIMessage)
        """
        trace_name = endpoint_name or "ProductAgent"

        langfuse_handler = LangfuseHandler(
            client_phone=client_phone,
            session_id=f"{client_phone}_{date.today()}",
            trace_name=trace_name,
            update_root=True,
        )

        # Инициализация AgentLogger
        agent_logger = get_agent_logger()
        agent_logger_callback = agent_logger.get_callback_handler(client_phone)

        use_langfuse_sdk = (
            LANGFUSE_SDK_AVAILABLE
            and settings.langfuse.langfuse_enabled
            and settings.langfuse.langfuse_public_key
        )

        try:
            if hasattr(self.llm, "temperature"):
                self.llm.temperature = DEFAULT_TEMPERATURE

            if use_langfuse_sdk:
                langfuse = get_client()

                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="product-agent-run",
                ) as span:
                    with propagate_attributes(
                        user_id=client_phone,
                        session_id=f"{client_phone}_{date.today()}",
                    ):
                        return await self._run_core(
                            user_input=user_input,
                            client_phone=client_phone,
                            trace_name=trace_name,
                            system_prompt=system_prompt,
                            langfuse_handler=langfuse_handler,
                            agent_logger_callback=agent_logger_callback,
                            use_hitl_on_error=True,
                            span=span,
                        )
            else:
                return await self._run_core(
                    user_input=user_input,
                    client_phone=client_phone,
                    trace_name=trace_name,
                    system_prompt=system_prompt,
                    langfuse_handler=langfuse_handler,
                    agent_logger_callback=agent_logger_callback,
                    use_hitl_on_error=False,
                    span=None,
                )

        except Exception as e:
            from src.services.ai.constants import ERROR_MESSAGE_AGENT_FAILED

            error_msg = ERROR_MESSAGE_AGENT_FAILED
            logger.error(
                f"[ProductAgent.run] Ошибка ProductAgent: {str(e)}",
                exc_info=True,
            )

            try:
                langfuse_handler.save_conversation_to_langfuse()
            except Exception as langfuse_error:
                logger.warning(
                    f"Не удалось сохранить ошибку в LangFuse: {langfuse_error}"
                )

            return error_msg

    async def _run_core(
        self,
        *,
        user_input: str,
        client_phone: str,
        trace_name: str,
        system_prompt: Optional[str],
        langfuse_handler: LangfuseHandler,
        agent_logger_callback: Any,
        use_hitl_on_error: bool,
        span: Optional[Any] = None,
    ) -> str:
        """Общий путь выполнения агента с подготовкой контекста и обработкой ошибок."""
        # 1. Подготовка промпта и истории
        chat_history: List[BaseMessage] = []

        if system_prompt:
            final_prompt = system_prompt

            if self.memory and is_memory_initialized(self.memory):
                try:
                    memory_vars = await self.memory.load_memory_variables(
                        {},
                        return_messages=True,
                    )
                    chat_history = memory_vars.get("history", [])
                except Exception as e:
                    logger.error(
                        "[ProductAgent] Ошибка загрузки памяти: %s",
                        e,
                        exc_info=True,
                    )
        else:
            base_prompt, system_vars, client_info, chat_history = (
                await self._load_prompt_and_context(
                    prompt_name=None,
                    client_phone=client_phone,
                )
            )
            final_prompt = base_prompt

        self.SYSTEM_PROMPT = final_prompt

        input_with_context = self._prepare_messages(user_input, chat_history)

        if span is not None:
            # Соответствует прежнему prep_span.update(...)
            span.update_trace(
                input={"original_query": user_input},
                output={
                    "processed_input": input_with_context[:200],
                    "prompt_length": len(final_prompt),
                    "chat_history_length": len(chat_history),
                },
            )

        # 2. Подготовка инструментов и конфигурации
        from src.tools.state_tools import set_photo_requirement

        context_tools = [set_photo_requirement]
        sql_tools = [generate_sql_from_text, execute_sql_query, get_table_schema]
        media_tools = [show_product_photos, send_pricelist]

        agent_tools = self.tools + sql_tools + media_tools + context_tools

        callbacks_list = [
            langfuse_handler,
            agent_logger_callback,
            StdOutCallbackHandler(),
        ]

        recursion_limit = AGENT_RECURSION_LIMIT

        config: RunnableConfig = {
            "callbacks": callbacks_list,
            "metadata": {
                "phone": client_phone,
                "user_id": client_phone,
                "trace_name": trace_name,
            },
            "run_name": trace_name,
            "tags": ["product_agent", "conversation", trace_name],
            "recursion_limit": recursion_limit,
        }

        messages: List[BaseMessage] = []
        if chat_history:
            messages.extend(chat_history)
        messages.append(HumanMessage(content=input_with_context))

        # 3. Выполнение агента с обработкой ошибок
        try:
            result = await self._execute_agent(
                messages,
                agent_tools,
                config,
                client_phone,
            )
        except asyncio.TimeoutError:
            error_msg = (
                f"Агент превысил максимальное время выполнения "
                f"({MAX_AGENT_EXECUTION_TIME} секунд)"
            )
            logger.error(f"[ProductAgent.run] Timeout агента: {error_msg}")

            if use_hitl_on_error:
                hitl_prompt = await self._get_hitl_prompt()
                if hitl_prompt:
                    return hitl_prompt

            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Ошибка при выполнении агента: {str(e)}"
            logger.error(
                f"[ProductAgent.run] Ошибка агента: {error_msg}",
                exc_info=True,
            )

            if use_hitl_on_error:
                hitl_prompt = await self._get_hitl_prompt()
                if hitl_prompt:
                    return hitl_prompt

            raise Exception(error_msg) from e

        # 4. Извлечение и постобработка ответа
        response_text = self._extract_response(result)

        if response_text is None or not response_text.strip():
            logger.warning(
                "[ProductAgent.run] Агент не смог сгенерировать ответ, "
                "используем HITL"
            )
            hitl_prompt = await self._get_hitl_prompt()
            if hitl_prompt:
                return hitl_prompt
            response_text = (
                "Извините, я не смог обработать ваш запрос. "
                "Пожалуйста, попробуйте переформулировать вопрос."
            )

        # 5. Логирование в Langfuse (для SDK-ветки сохраняем совместимое поведение)
        if span is not None:
            span.update_trace(
                input={"original_query": user_input},
                output={"response": response_text[:500]},
            )

        # 6. Сохранение в память и Langfuse
        await self._save_to_memory(
            user_input,
            response_text,
            chat_history,
            client_phone,
        )

        langfuse_handler.save_conversation_to_langfuse()

        return response_text