"""ProductAgent - агент для работы с продуктами и продажами.

Использует LangChain AgentExecutor для обработки запросов пользователей
с использованием tools, памяти и профиля клиента.
"""

from __future__ import annotations

from typing import Any, List, Optional
import hashlib
import logging
from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_react_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManager
from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langchain_core.tracers import LangChainTracer

from src.config.settings import settings
from src.config.langchain_settings import LangChainSettings
from src.config.constants import (
    DEFAULT_TEMPERATURE,
    MAX_AGENT_ITERATIONS,
    MAX_AGENT_EXECUTION_TIME,
)
from src.utils.langfuse_handler import LangfuseHandler
from .base_agent import BaseAgent
from agents.tools import (
    enhance_user_product_query,
    show_product_photos,
    get_client_profile,
    text_to_sql_products,
    get_random_products,
)
from src.utils.prompts import (
    get_prompt,
    get_all_system_values,
    build_prompt_with_context,
)

logger = logging.getLogger(__name__)

langchain_settings = LangChainSettings()


class ProductAgent(BaseAgent):
    """Агент для обработки запросов пользователей о товарах и продажах.

    Использует AgentExecutor с tools для поиска товаров, отправки фото
    и получения профиля клиента.
    """

    DEFAULT_SYSTEM_PROMPT = """Ты профессиональный, дружелюбный и отзывчивый менеджер по продажам мясной продукции. 😊

Твоя задача:
- Помогать клиентам найти подходящие товары из ассортимента
- Предоставлять детальную информацию о товарах (цена, вес, упаковка, поставщик)
- Отправлять фотографии товаров по запросу клиента
- Учитывать профиль клиента при рекомендациях
- Быть вежливым, дружелюбным и человечным в общении

ВАЖНО - стиль общения:
- Используй смайлики уместно и естественно (😊, 👍, 🥩, 🔥, ⚡, 💪 и т.д.)
- Пиши так, как общается живой человек - тепло и дружелюбно
- Будь позитивным и энергичным, но не навязчивым
- Используй разговорные фразы: "Конечно!", "С удовольствием!", "Отлично!", "Замечательно!"
- Если не нашел товары, предлагай альтернативы с позитивным настроем

Используй доступные инструменты для:
- Поиска товаров по запросу клиента (enhance_user_product_query) - используй для запросов с текстовыми критериями (названия товаров, поставщики, регионы, типы мяса)
- Поиска товаров по числовым условиям (text_to_sql_products) - ОБЯЗАТЕЛЬНО используй для запросов с числовыми условиями (цена, вес, скидка, минимальный заказ)
- Отправки фотографий товаров (show_product_photos)
- Получения информации о профиле клиента (get_client_profile)

КРИТИЧЕСКИ ВАЖНО - ПРАВИЛА ВЫБОРА ИНСТРУМЕНТОВ:
ПЕРВЫМ ДЕЛОМ проверяй наличие ЧИСЛОВЫХ условий в запросе!

- Если запрос содержит ЧИСЛОВЫЕ условия про ЦЕНУ ("цена меньше 80", "дешевле 100 рублей", "цена от 50 до 200", "стоимость меньше X") - ВСЕГДА используй text_to_sql_products!
- Если запрос содержит ЧИСЛОВЫЕ условия про ВЕС ("вес больше 5 кг", "минимальный заказ меньше 10") - ВСЕГДА используй text_to_sql_products!
- Если запрос содержит ЧИСЛОВЫЕ условия про СКИДКУ ("скидка больше 15%", "скидка от 10 до 20") - ВСЕГДА используй text_to_sql_products!
- Если запрос содержит КОМБИНАЦИЮ числовых условий ("цена меньше 100 и скидка больше 10%") - ВСЕГДА используй text_to_sql_products!
- Для запросов про поставщиков БЕЗ чисел ("Что есть из продукции Мироторг", "товары от X") - используй enhance_user_product_query
- Для запросов про регионы БЕЗ чисел ("мясо из Сибири", "товары из региона Z") - используй enhance_user_product_query
- Для запросов с названиями товаров БЕЗ чисел ("говядина", "стейки", "полуфабрикаты") - используй enhance_user_product_query
- НИКОГДА не используй text_to_sql_products для запросов БЕЗ числовых условий!

Всегда старайся помочь клиенту найти именно то, что он ищет, и будь максимально дружелюбным! 😊"""

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
        langchain_settings.setup_langsmith_tracing()

        callbacks = None
        if (
            langchain_settings.langsmith_tracing_enabled
            and langchain_settings.langsmith_api_key
        ):
            langsmith_client = Client(api_key=langchain_settings.langsmith_api_key)
            langsmith_tracer = LangChainTracer(
                project_name=langchain_settings.langsmith_project_name,
                client=langsmith_client,
            )
            callbacks = CallbackManager([langsmith_tracer])

        if llm is None:
            llm = ChatOpenAI(
                model=settings.openrouter.model_id,
                openai_api_key=settings.openrouter.openrouter_api_key,
                openai_api_base=settings.openrouter.base_url,
                temperature=DEFAULT_TEMPERATURE,
            )

        if tools is None:
            tools = [
                enhance_user_product_query,
                show_product_photos,
                get_client_profile,
                text_to_sql_products,
                get_random_products,
            ]

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.agent_type = agent_type
        self._agent_executor: Optional[AgentExecutor] = None
        self._callbacks = callbacks
        self.SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT
        self._last_prompt_hash: Optional[str] = None

    def _create_agent_executor(self, callbacks=None) -> AgentExecutor:
        """Создаёт AgentExecutor с промптом и инструментами.

        Args:
            callbacks: Callback'и для передачи в AgentExecutor (опционально)

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
            callbacks=callbacks,
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

        Returns:
            Строка с ответом агента
        """
        logger.info(
            f"[ProductAgent.run] Начало выполнения для {client_phone}, topic: {topic}"
        )

        from datetime import date

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
            db_prompt = None
            if topic:
                try:
                    db_prompt = await get_prompt(topic)
                    if db_prompt:
                        logger.info(f"Загружен промпт из БД для topic '{topic}'")
                except Exception as e:
                    logger.warning(
                        f"Не удалось загрузить промпт для topic '{topic}': {e}"
                    )

            system_vars = {}
            try:
                system_vars = await get_all_system_values()
                if system_vars:
                    logger.info(f"Загружено системных переменных: {len(system_vars)}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить системные переменные: {e}")

            profile_context = ""
            try:
                profile_result = await get_client_profile.ainvoke(
                    {"phone": client_phone}
                )
                if profile_result and "не найден" not in profile_result.lower():
                    profile_context = profile_result
            except Exception as e:
                logger.warning(f"Не удалось загрузить профиль клиента: {e}")

            final_prompt = None

            if db_prompt:
                enhanced_prompt = build_prompt_with_context(
                    base_prompt=db_prompt,
                    client_info=profile_context if profile_context else None,
                    system_vars=system_vars if system_vars else None,
                )
                final_prompt = enhanced_prompt
            elif system_vars:
                system_vars_text = "\n".join(
                    [f"{k}: {v}" for k, v in system_vars.items()]
                )
                final_prompt = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nСистемные переменные:\n{system_vars_text}"
            else:
                final_prompt = self.DEFAULT_SYSTEM_PROMPT

            prompt_hash = hashlib.md5(final_prompt.encode()).hexdigest()
            if self._last_prompt_hash != prompt_hash:
                logger.info("Промпт изменился, пересоздаем AgentExecutor")
                self.SYSTEM_PROMPT = final_prompt
                self._last_prompt_hash = prompt_hash
                self._agent_executor = None

            chat_history: List[BaseMessage] = []
            if self.memory is not None:
                try:
                    memory_vars = await self.memory.load_memory_variables(
                        {}, return_messages=True
                    )
                    chat_history = memory_vars.get("history", [])
                except Exception as e:
                    logger.warning(f"Не удалось загрузить память: {e}")
                    chat_history = []

            input_with_context = user_input
            full_prompt_parts = ["=== ПОЛНЫЙ ПРОМПТ К LLM ===\n"]
            full_prompt_parts.append(f"System:\n{self.SYSTEM_PROMPT}\n")

            if chat_history:
                full_prompt_parts.append(
                    f"Chat History ({len(chat_history)} сообщений):"
                )
                for i, msg in enumerate(chat_history, 1):
                    if isinstance(msg, HumanMessage):
                        full_prompt_parts.append(f"  [{i}] Human: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        full_prompt_parts.append(f"  [{i}] AI: {msg.content}")
                    elif isinstance(msg, SystemMessage):
                        full_prompt_parts.append(f"  [{i}] System: {msg.content}")
            else:
                full_prompt_parts.append("Chat History: (пусто)")

            if is_init_message:
                full_prompt_parts.append(
                    f"\nInit Message (System):\n{input_with_context}\n"
                )
            else:
                full_prompt_parts.append(f"\nUser Input:\n{input_with_context}\n")
            full_prompt_parts.append("=" * 50)

            try:
                from langchain_core.callbacks import CallbackManager
                from langchain_core.callbacks.stdout import StdOutCallbackHandler

                callbacks_list = []

                if self._callbacks:
                    if hasattr(self._callbacks, "handlers"):
                        callbacks_list.extend(self._callbacks.handlers)
                    elif isinstance(self._callbacks, list):
                        callbacks_list.extend(self._callbacks)
                    else:
                        callbacks_list.append(self._callbacks)

                callbacks_list.append(langfuse_handler)

                stdout_handler = StdOutCallbackHandler()
                callbacks_list.append(stdout_handler)

                logger.info(
                    f"[ProductAgent.run] Подготовлено {len(callbacks_list)} callbacks: "
                    f"{[type(cb).__name__ for cb in callbacks_list]}"
                )

                combined_callbacks = CallbackManager(callbacks_list)
                agent_executor = self._create_agent_executor(
                    callbacks=combined_callbacks
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
                logger.error(f"Ошибка AgentExecutor: {error_msg}", exc_info=True)
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
                    logger.warning(f"Не удалось сохранить в память: {e}")

            langfuse_handler.save_conversation_to_langfuse()

            tools_list = sorted(list(langfuse_handler.used_tools))
            if tools_list:
                tool_type_map = {
                    "enhance_user_product_query": "VECTOR SEARCH",
                    "text_to_sql_products": "TEXT-TO-SQL",
                    "show_product_photos": "PHOTO SENDER",
                    "get_client_profile": "CLIENT PROFILE",
                }
                tools_summary = []
                for tool_name in tools_list:
                    call_count = sum(
                        1
                        for tc in langfuse_handler.tool_calls
                        if tc.get("tool_name") == tool_name
                    )
                    tool_type = tool_type_map.get(tool_name, "TOOL")
                    tools_summary.append(f"{tool_type} {tool_name}({call_count}x)")
                logger.info(
                    f"Завершение для {client_phone}: использовано {len(tools_list)} инструментов: {', '.join(tools_summary)}"
                )

            return response_text

        except Exception as e:
            error_msg = (
                f"Ой, что-то пошло не так 😔. Попробуйте написать еще раз, пожалуйста!"
            )
            logger.error(f"Ошибка ProductAgent: {str(e)}", exc_info=True)

            try:
                langfuse_handler.save_conversation_to_langfuse()
            except Exception as langfuse_error:
                logger.warning(
                    f"Не удалось сохранить ошибку в LangFuse: {langfuse_error}"
                )

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
