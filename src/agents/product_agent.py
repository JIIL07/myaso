"""ProductAgent - агент для работы с продуктами и продажами.

Использует LangChain AgentExecutor для обработки запросов пользователей
с использованием tools, памяти и профиля клиента.
"""

from __future__ import annotations

from typing import Any, List, Optional
import logging
import hashlib
from datetime import date
from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_react_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.runnables import RunnableConfig

from src.config.settings import settings
from src.config.constants import (
    DEFAULT_TEMPERATURE,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_EXECUTION_TIME,
)
from src.utils.callbacks.langfuse_callback import LangfuseHandler
from .base_agent import BaseAgent
from .tools import (
    vector_search,
    get_client_profile,
    generate_sql_from_text,
    execute_sql_request,
    get_random_products,
)
from .tools.media_tools import create_media_tools
from src.utils.prompts import (
    get_prompt,
    get_all_system_values,
    build_prompt_with_context,
)

logger = logging.getLogger(__name__)


class ProductAgent(BaseAgent):
    """Агент для обработки запросов пользователей о товарах и продажах.

    Использует AgentExecutor с tools для поиска товаров, отправки фото
    и получения профиля клиента.
    """

    DEFAULT_SYSTEM_PROMPT = """
==========================================================================================================
ПОКАЗЫВАЙ ВСЕ НАЙДЕННЫЕ ТОВАРЫ
==========================================================================================================

ВАЖНО: Когда инструменты поиска (vector_search, execute_sql_request, get_random_products) 
возвращают список товаров, ты ДОЛЖЕН показать клиенту ВСЕ найденные товары БЕЗ исключений!

ПРАВИЛА:
1. Если инструмент вернул "Найдено товаров: 50" - покажи клиенту ВСЕ 50 товаров
2. НЕ сокращай список товаров в своем ответе - покажи ВСЕ товары из ответа инструмента
3. НЕ показывай только "первые несколько" или "примеры" - покажи ВСЕ
4. Если в ответе инструмента есть предупреждение "⚠️ В базе данных есть ещё товары" - обязательно упомяни это клиенту
5. Используй тот же компактный формат, что и в ответе инструмента (каждая строка = один товар)
6. Если список длинный (50 товаров), это нормально - покажи его полностью, клиент должен видеть весь ассортимент

ПРИМЕР ПРАВИЛЬНОГО ПОВЕДЕНИЯ:
Инструмент вернул: "Найдено товаров: 50\n\nТовар 1 | Поставщик: X | 100₽/кг\nТовар 2 | Поставщик: Y | 200₽/кг\n..."
Твой ответ клиенту: покажи ВСЕ 50 товаров в том же компактном формате, включая предупреждение о дополнительных товарах если оно есть

ПРИМЕР НЕПРАВИЛЬНОГО ПОВЕДЕНИЯ:
Инструмент вернул 50 товаров, а ты показываешь только 10 - ЭТО НЕПРАВИЛЬНО!
Инструмент вернул предупреждение о дополнительных товарах, а ты его не упомянул - ЭТО НЕПРАВИЛЬНО!

==========================================================================================================
РАСЧЕТ ФИНАЛЬНОЙ ЦЕНЫ
==========================================================================================================

ВАЖНО: Всегда показывай клиенту ФИНАЛЬНУЮ цену (final_price_kg), а не order_price_kg!

Правила расчета финальной цены (используются значения из SYS VARIABLES):
1. Если order_price_kg < 100: final_price_kg = order_price_kg + наценка (из SYS VARIABLES)
2. Если order_price_kg >= 100: final_price_kg = order_price_kg * коэффициент (из SYS VARIABLES) + order_price_kg

Всегда выводи final_price_kg в ответах клиенту, а не order_price_kg!
Системные переменные для расчета цен находятся в блоке SYS VARIABLES выше."""

    def __init__(
        self,
        *,
        llm: Optional[Any] = None,
        retriever: Optional[Any] = None,
        memory: Optional[Any] = None,
        tools: Optional[List[Any]] = None,
        agent_type: str = "openai-tools",
        **kwargs: Any,
    ) -> None:
        """Инициализация ProductAgent.

        Args:
            llm: LangChain LLM модель (если None, создаётся ChatOpenAI)
            retriever: Векторный ретривер (опционально, для будущего использования)
            memory: Память диалога (BaseChatMessageHistory)
            tools: Список инструментов (если None, используются стандартные)
            agent_type: Тип агента - "openai-tools" или "zero-shot-react-description"
            **kwargs: Дополнительные параметры для BaseAgent
        """
        if llm is None:
            llm = ChatOpenAI(
                model=settings.openrouter.model_id,
                openai_api_key=settings.openrouter.openrouter_api_key,
                openai_api_base=settings.openrouter.base_url,
                temperature=DEFAULT_TEMPERATURE,
            )

        if tools is None:
            tools = [
                vector_search,
                get_client_profile,
                generate_sql_from_text,
                execute_sql_request,
                get_random_products,
            ]

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.agent_type = agent_type
        self.SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT
        self._executor_cache: dict[str, AgentExecutor] = {}
        self._cached_prompt_hash: Optional[str] = None

    def _get_prompt_hash(self, system_prompt: str) -> str:
        """Вычисляет хеш промпта для кэширования.
        """
        return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()

    def _create_agent_executor(self) -> AgentExecutor:
        """Создаёт AgentExecutor с промптом и инструментами.

        Returns:
            AgentExecutor для выполнения агента
        """
        system_prompt = self.SYSTEM_PROMPT

        if self.agent_type == "openai-tools":
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_react_agent(self.llm, self.tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=MAX_AGENT_ITERATIONS,
            max_execution_time=MAX_AGENT_EXECUTION_TIME,
        )

        return agent_executor

    def _get_agent_executor(self) -> AgentExecutor:
        """Получает AgentExecutor из кэша или создает новый.

        Кэширует AgentExecutor по хешу текущего SYSTEM_PROMPT.
        Если промпт изменился, создает новый executor.

        Returns:
            AgentExecutor для выполнения агента
        """
        current_prompt_hash = self._get_prompt_hash(self.SYSTEM_PROMPT)

        if current_prompt_hash != self._cached_prompt_hash or current_prompt_hash not in self._executor_cache:
            if current_prompt_hash != self._cached_prompt_hash:
                self._executor_cache.clear()

            executor = self._create_agent_executor()
            self._executor_cache[current_prompt_hash] = executor
            self._cached_prompt_hash = current_prompt_hash

        return self._executor_cache[current_prompt_hash]

    async def run(
        self,
        user_input: str,
        client_phone: str,
        topic: Optional[str] = None,
        is_init_message: bool = False,
        endpoint_name: Optional[str] = None,
    ) -> str:
        """Запускает агента для обработки запроса пользователя.

        Args:
            user_input: Текст запроса пользователя
            client_phone: Номер телефона клиента
            topic: Тема диалога для загрузки промпта из БД (опционально)
            is_init_message: Если True, не сохраняет user_input в память (для системных промптов init)

        Returns:
            Строка с ответом агента
        """
        trace_name = endpoint_name or "AgentExecutor"

        langfuse_handler = LangfuseHandler(
            client_phone=client_phone,
            session_id=f"{client_phone}_{date.today()}",
            trace_name=trace_name,
        )

        logger.info(
            f"[ProductAgent.run] LangfuseHandler создан для {client_phone}, "
            f"type={type(langfuse_handler).__name__}"
        )

        try:
            logger.info(
                f"[ProductAgent.run] Начало обработки запроса для {client_phone}, topic: {topic}"
            )

            db_prompt = None
            if topic:
                try:
                    db_prompt = await get_prompt(topic)
                except Exception as e:
                    logger.error(
                        f"[ProductAgent.run] Не удалось загрузить промпт для topic '{topic}': {e}"
                    )

            system_vars = {}
            try:
                system_vars = await get_all_system_values()
            except Exception as e:
                logger.error(f"[ProductAgent.run] Не удалось загрузить системные переменные: {e}")


            profile_context = ""

            if db_prompt:
                base_prompt = db_prompt + f"{self.SYSTEM_PROMPT}"
            else:
                base_prompt = self.DEFAULT_SYSTEM_PROMPT

            final_prompt = build_prompt_with_context(
                base_prompt=base_prompt,
                client_info=profile_context if profile_context else None,
                system_vars=system_vars if system_vars else None,
            )
            self.SYSTEM_PROMPT = final_prompt

            chat_history: List[BaseMessage] = []
            if self.memory is not None:
                try:
                    if not hasattr(self.memory, 'async_initialized') or not self.memory.async_initialized:
                        logger.warning(f"[ProductAgent.run] Память не инициализирована для {client_phone}, пропускаем загрузку истории")
                        chat_history = []
                    else:
                        memory_vars = await self.memory.load_memory_variables(
                            {}, return_messages=True
                        )
                        chat_history = memory_vars.get("history", [])
                        logger.info(f"[ProductAgent.run] Загружено {len(chat_history)} сообщений из памяти для {client_phone}")
                except Exception as e:
                    logger.error(f"[ProductAgent.run] Не удалось загрузить память: {e}", exc_info=True)
                    chat_history = []

            media_tools = create_media_tools(client_phone, is_init_message=is_init_message)
            other_tools = [
                vector_search,
                get_client_profile,
                generate_sql_from_text,
                execute_sql_request,
                get_random_products,
            ]
            all_tools = media_tools + other_tools

            logger.info(f"[ProductAgent.run] Created tools for client: {client_phone}")

            input_with_context = user_input

            try:
                callbacks_list = []
                callbacks_list.append(langfuse_handler)

                stdout_handler = StdOutCallbackHandler()
                callbacks_list.append(stdout_handler)

                logger.info(
                    f"[ProductAgent.run] Подготовлено {len(callbacks_list)} callbacks: "
                    f"{[type(cb).__name__ for cb in callbacks_list]}"
                )

                combined_callbacks = CallbackManager(callbacks_list)

                system_prompt = self.SYSTEM_PROMPT

                if self.agent_type == "openai-tools":
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            MessagesPlaceholder(variable_name="agent_scratchpad"),
                        ]
                    )
                    agent = create_openai_tools_agent(self.llm, all_tools, prompt)
                else:
                    prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt),
                            MessagesPlaceholder(variable_name="chat_history"),
                            ("human", "{input}"),
                            MessagesPlaceholder(variable_name="agent_scratchpad"),
                        ]
                    )
                    agent = create_react_agent(self.llm, all_tools, prompt)

                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=all_tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=MAX_AGENT_ITERATIONS,
                    max_execution_time=MAX_AGENT_EXECUTION_TIME,
                    callbacks=combined_callbacks,
                )

                config: RunnableConfig = {
                    "callbacks": callbacks_list,
                    "metadata": {
                        "phone": client_phone,
                        "user_id": client_phone,
                        "trace_name": trace_name,
                    },
                    "run_name": trace_name,
                    "tags": ["product_agent", "conversation", trace_name],
                }

                result = await agent_executor.ainvoke(
                    {
                        "input": input_with_context,
                        "chat_history": chat_history,
                    },
                    config=config,
                )
            except Exception as e:
                error_msg = f"Ошибка при выполнении агента: {str(e)}"
                logger.error(f"[ProductAgent.run] Ошибка AgentExecutor: {error_msg}", exc_info=True)
                raise Exception(error_msg) from e

            response_text = result.get("output", "")
            if not response_text:
                response_text = "Упс, что-то пошло не так 😅. Попробуйте переформулировать запрос, и я обязательно помогу!"

            if self.memory is not None:
                try:
                    if not is_init_message:
                        await self.memory.add_messages(
                            [HumanMessage(content=user_input)]
                        )
                        await self.memory.add_messages(
                            [AIMessage(content=response_text)]
                        )
                except Exception as e:
                    logger.warning(f"[ProductAgent.run] Не удалось сохранить в память: {e}", exc_info=True)

            langfuse_handler.save_conversation_to_langfuse()

            return response_text

        except Exception as e:
            error_msg = (
                f"Ой, что-то пошло не так 😔. Попробуйте написать еще раз, пожалуйста!"
            )
            logger.error(f"[ProductAgent.run] Ошибка ProductAgent: {str(e)}", exc_info=True)

            try:
                langfuse_handler.save_conversation_to_langfuse()
            except Exception as langfuse_error:
                logger.warning(
                    f"Не удалось сохранить ошибку в LangFuse: {langfuse_error}"
                )

            logger.info(f"[ProductAgent.run] Завершение обработки запроса для {client_phone} с ошибкой")
            return error_msg

    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        """Собирает промпт для модели.

        Args:
            user_input: Входной запрос пользователя
            **kwargs: Дополнительные параметры

        Returns:
            Строка с промптом
        """
        return user_input

    def _create_tools(self) -> List[Any]:
        """Создаёт и возвращает список инструментов.

        Returns:
            Список инструментов агента
        """
        return self.tools
