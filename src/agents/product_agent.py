"""ProductAgent - агент для работы с продуктами и каталогом.

Использует LangChain AgentExecutor для обработки запросов пользователей
с использованием tools для поиска товаров через семантический поиск и SQL фильтрацию.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, List, Optional

from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_react_agent,
)
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.config.constants import (
    DEFAULT_TEMPERATURE,
    MAX_AGENT_EXECUTION_TIME,
    MAX_AGENT_ITERATIONS,
)
from src.config.settings import settings
from src.database.queries.clients_queries import get_client_is_friend
from src.utils.callbacks.langfuse_callback import LangfuseHandler
from src.utils.callbacks.reasoning_logger import ReasoningLogger
from src.utils.prompts import (
    build_prompt_with_context,
    get_all_system_values,
    get_prompt,
)

from .base_agent import BaseAgent
from .tools.client_tools import get_client_profile
from .tools.media_tools import create_media_tools
from .tools.product_tools import get_random_products, vector_search
from .tools.sql_tools import create_sql_tools

logger = logging.getLogger(__name__)


def is_greeting_message(message: str) -> bool:
    """Проверяет, содержит ли сообщение приветствие.
    
    Args:
        message: Текст сообщения
        
    Returns:
        True если сообщение содержит приветствие, False иначе
    """
    if not message:
        return False
    
    message_lower = message.lower().strip()
    
    greetings = [
        "привет", "здравствуй", "здравствуйте", "добрый день", "добрый вечер",
        "доброе утро", "доброй ночи", "доброго дня", "доброго вечера",
        "доброго утра", "здорово", "салют", "хай", "hi", "hello",
        "доброго времени суток", "приветствую", "добро пожаловать"
    ]
    
    for greeting in greetings:
        if message_lower.startswith(greeting) or f" {greeting} " in f" {message_lower} ":
            return True
    
    return False


class ProductAgent(BaseAgent):
    """Агент для обработки запросов пользователей о товарах и каталоге.

    Использует AgentExecutor с tools для поиска товаров через:
    - vector_search для семантического поиска
    - t + execute_sql_query для фильтрации по параметрам
    - get_random_products как fallback
    """

    DEFAULT_SYSTEM_PROMPT = ""

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
            try:
                if not hasattr(settings, 'openrouter'):
                    raise ValueError("settings.openrouter не найден")
                
                if not settings.openrouter.model_id:
                    raise ValueError("settings.openrouter.model_id не установлен")
                
                if not settings.openrouter.openrouter_api_key:
                    raise ValueError("settings.openrouter.openrouter_api_key не установлен")
                
                llm = ChatOpenAI(
                    model=settings.openrouter.model_id,
                    openai_api_key=settings.openrouter.openrouter_api_key,
                    openai_api_base=settings.openrouter.base_url,
                    temperature=DEFAULT_TEMPERATURE,
                )
                logger.info(
                    f"[ProductAgent] LLM инициализирован: "
                    f"model={settings.openrouter.model_id}, "
                    f"base_url={settings.openrouter.base_url}, "
                    f"temperature={DEFAULT_TEMPERATURE}"
                )
            except Exception as e:
                logger.error(
                    f"[ProductAgent] Ошибка инициализации LLM: {e}",
                    exc_info=True
                )
                raise ValueError(f"Не удалось инициализировать LLM: {e}") from e

        if tools is None:
            tools = [
                get_client_profile,
                vector_search,
                get_random_products,
            ]

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.agent_type = agent_type
        self.SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT

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

    def build_prompt(self, user_input: str, **kwargs: Any) -> str:
        """Собирает промпт для модели (публичный метод для обратной совместимости).

        Args:
            user_input: Входной запрос пользователя
            **kwargs: Дополнительные параметры

        Returns:
            Строка с промптом
        """
        return self._build_prompt(user_input, **kwargs)

    def create_tools(self) -> List[Any]:
        """Создаёт и возвращает список инструментов (публичный метод для обратной совместимости).

        Returns:
            Список инструментов агента
        """
        return self._create_tools()

    def create_agent_executor(
        self, callbacks: Optional[List[Any]] = None, tools: Optional[List[Any]] = None
    ) -> AgentExecutor:
        """Создаёт AgentExecutor с промптом и инструментами.

        Args:
            callbacks: Список callbacks для AgentExecutor
            tools: Список инструментов (если None, используются self.tools)

        Returns:
            AgentExecutor для выполнения агента
        """
        system_prompt = self.SYSTEM_PROMPT
        agent_tools = tools or self.tools

        if self.agent_type == "openai-tools":
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_openai_tools_agent(self.llm, agent_tools, prompt)
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_react_agent(self.llm, agent_tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=agent_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=MAX_AGENT_ITERATIONS,
            max_execution_time=MAX_AGENT_EXECUTION_TIME,
            callbacks=None,
        )

        return agent_executor

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
            endpoint_name: Имя endpoint для трейсинга

        Returns:
            Строка с ответом агента
        """
        trace_name = endpoint_name or "ProductAgent"

        langfuse_handler = LangfuseHandler(
            client_phone=client_phone,
            session_id=f"{client_phone}_{date.today()}",
            trace_name=trace_name,
        )

        try:
            logger.info(
                f"[ProductAgent.run] Начало обработки запроса для {client_phone}, topic: {topic}, "
                f"user_input (полный): '{user_input}'"
            )

            db_prompt = None
            if topic:
                try:
                    db_prompt = await get_prompt(topic)
                    if db_prompt:
                        logger.info(
                            f"[ProductAgent.run] Загружен промпт из БД для topic '{topic}': "
                            f"длина={len(db_prompt)} символов, первые 200 символов: '{db_prompt[:200]}...'"
                        )
                    else:
                        logger.warning(f"[ProductAgent.run] Промпт для topic '{topic}' не найден в БД")
                except Exception as e:
                    logger.error(
                        f"[ProductAgent.run] Не удалось загрузить промпт для topic '{topic}': {e}"
                    )

            system_vars = {}
            try:
                system_vars = await get_all_system_values()
            except Exception as e:
                logger.error(f"[ProductAgent.run] Не удалось загрузить системные переменные: {e}")

            if db_prompt:
                base_prompt = db_prompt + f"\n\n{self.DEFAULT_SYSTEM_PROMPT}"
                logger.info(
                    f"[ProductAgent.run] Промпт из БД объединен с системным промптом. "
                    f"Общая длина base_prompt: {len(base_prompt)} символов"
                )
            else:
                base_prompt = self.DEFAULT_SYSTEM_PROMPT
                logger.info(
                    f"[ProductAgent.run] Используется только системный промпт (промпт из БД не загружен). "
                    f"Длина: {len(base_prompt)} символов"
                )

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

            client_is_friend = False
            try:
                client_is_friend = await get_client_is_friend(client_phone)
                logger.info(f"[ProductAgent.run] Клиент {client_phone}: is_it_friend={client_is_friend}")
            except Exception as e:
                logger.error(f"[ProductAgent.run] Не удалось получить статус дружбы клиента: {e}", exc_info=True)

            is_second_message = False
            client_greeted = is_greeting_message(user_input)
            
            if len(chat_history) == 1:
                if isinstance(chat_history[0], AIMessage):
                    is_second_message = True
                    logger.info(f"[ProductAgent.run] Определено как второе сообщение в разговоре (история: 1 сообщение от ассистента)")
            elif len(chat_history) == 2:
                if isinstance(chat_history[0], AIMessage) and isinstance(chat_history[1], HumanMessage):
                    is_second_message = True
                    logger.info(f"[ProductAgent.run] Определено как второе сообщение в разговоре (история: приветствие + ответ)")

            client_info_parts = []
            client_info_parts.append(f"Номер телефона: {client_phone}")
            client_info_parts.append(f"Статус дружбы (it_is_friend): {client_is_friend}")
            if client_is_friend:
                client_info_parts.append("ОБРАЩЕНИЕ: Используй 'ты' (неформальное общение)")
            else:
                client_info_parts.append("ОБРАЩЕНИЕ: Используй 'вы' (формальное общение)")
            
            client_info = "\n".join(client_info_parts)

            final_prompt = build_prompt_with_context(
                base_prompt=base_prompt,
                client_info=client_info,
                system_vars=system_vars if system_vars else None,
            )
            self.SYSTEM_PROMPT = final_prompt
            
            logger.info(
                f"[ProductAgent.run] Финальный SYSTEM_PROMPT собран и установлен для агента. "
                f"Длина: {len(final_prompt)} символов. "
                f"Содержит промпт из БД: {'ДА' if db_prompt else 'НЕТ'}. "
                f"Первые 300 символов: '{final_prompt[:300]}...'"
            )

            context_parts = []
            if client_greeted:
                if is_second_message:
                    context_parts.append("ВАЖНО: Это второе сообщение, но клиент поздоровался с тобой. Поздоровайся в ответ, затем продолжай общение.")
                else:
                    context_parts.append("ВАЖНО: Клиент поздоровался с тобой. Поздоровайся в ответ, затем продолжай общение.")
            elif is_second_message:
                context_parts.append("ВАЖНО: Это второе сообщение в разговоре. НЕ используй приветствие, сразу переходи к делу.")
            
            input_with_context = user_input
            if context_parts:
                input_with_context = user_input + "\n\n" + "\n".join(context_parts)
            
            logger.info(
                f"[ProductAgent.run] Финальный запрос для агента (input_with_context): '{input_with_context}'"
            )

            sql_tools = create_sql_tools(is_init_message=is_init_message)
            media_tools = create_media_tools(client_phone=client_phone, is_init_message=is_init_message)
            agent_tools = self.tools + sql_tools + media_tools

            try:
                callbacks_list = []
                callbacks_list.append(langfuse_handler)

                stdout_handler = StdOutCallbackHandler()
                callbacks_list.append(stdout_handler)

                reasoning_logger = ReasoningLogger(client_phone=client_phone)
                callbacks_list.append(reasoning_logger)

                logger.info(
                    f"[ProductAgent.run] Подготовлено {len(callbacks_list)} callbacks: "
                    f"{[type(cb).__name__ for cb in callbacks_list]}"
                )

                agent_executor = self.create_agent_executor(callbacks=None, tools=agent_tools)

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
            
            if result:
                intermediate_steps = result.get("intermediate_steps", [])
                steps_count = len(intermediate_steps) if intermediate_steps else 0
                
                # Получаем сводку от reasoning_logger
                try:
                    summary = reasoning_logger.get_summary()
                    
                    # Логируем только важную информацию без дублирования
                    if steps_count == 0:
                        logger.warning(
                            f"[ProductAgent.run] ⚠️ НЕТ ПРОМЕЖУТОЧНЫХ ШАГОВ для {client_phone}: "
                            f"агент ответил БЕЗ вызова инструментов! "
                            f"user_input='{user_input[:200]}', "
                            f"LLM вызовов={summary['llm_calls']}, "
                            f"без инструментов={summary['llm_calls_without_tools']}"
                        )
                    else:
                        # Логируем какие инструменты были вызваны
                        tool_names_list = []
                        for step in intermediate_steps:
                            if len(step) >= 2:
                                action = step[0]
                                tool_name = "unknown"
                                if hasattr(action, "tool"):
                                    tool_name = action.tool
                                elif isinstance(action, dict):
                                    tool_name = action.get("tool", "unknown")
                                tool_names_list.append(tool_name)
                        
                        logger.info(
                            f"[ProductAgent.run] ✅ Запрос обработан для {client_phone}: "
                            f"использовано {steps_count} инструмент(ов) {tool_names_list}, "
                            f"LLM вызовов={summary['llm_calls']}, "
                            f"response_length={len(response_text)}"
                        )
                except Exception as e:
                    logger.debug(f"[ProductAgent.run] Не удалось получить reasoning summary: {e}")
                    # Fallback логирование без reasoning_logger
                    logger.info(
                        f"[ProductAgent.run] Запрос обработан: "
                        f"user_input={user_input[:100]}, "
                        f"steps={steps_count}, "
                        f"response_length={len(response_text)}"
                    )

            if self.memory is not None:
                try:
                    if not hasattr(self.memory, 'async_initialized') or not self.memory.async_initialized:
                        logger.warning(f"[ProductAgent.run] Память не инициализирована для {client_phone}, пропускаем сохранение")
                    elif not is_init_message:
                        logger.info(f"[ProductAgent.run] Сохранение сообщений в память для {client_phone}: user_input и response")
                        await self.memory.add_messages(
                            [HumanMessage(content=user_input)]
                        )
                        await self.memory.add_messages(
                            [AIMessage(content=response_text)]
                        )
                        logger.info(f"[ProductAgent.run] Сообщения успешно сохранены в память для {client_phone}")
                    else:
                        logger.info(f"[ProductAgent.run] Сохранение только ответа агента (init_message) для {client_phone}")
                        await self.memory.add_messages(
                            [AIMessage(content=response_text)]
                        )
                        logger.info(f"[ProductAgent.run] Ответ агента успешно сохранен в память для {client_phone}")
                except Exception as e:
                    logger.error(f"[ProductAgent.run] Не удалось сохранить в память для {client_phone}: {e}", exc_info=True)

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