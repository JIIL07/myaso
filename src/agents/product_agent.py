"""ProductAgent - агент для работы с продуктами и продажами.

Использует LangChain AgentExecutor для обработки запросов пользователей
с использованием tools, памяти и профиля клиента.
"""

from __future__ import annotations

from typing import Any, List, Optional
import logging
import hashlib
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
from src.utils.callbacks.langfuse_handler import LangfuseHandler
from .base_agent import BaseAgent
from .tools import (
    vector_search,
    show_product_photos,
    get_client_profile,
    generate_sql_from_text,
    execute_sql_request,
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

    DEFAULT_SYSTEM_PROMPT = """Ты - Эдуард, дружелюбный и энергичный помощник по продажам мясной продукции.

==========================================================================================================
ПРОФИЛЬ ПЕРСОНАЖА
==========================================================================================================

Имя: Эдуард
Характер: Дружелюбный, энергичный, позитивный, профессиональный, но неформальный
Стиль общения:
- Используй дружелюбный тон, но оставайся профессиональным
- Будь позитивным и энергичным
- Используй эмоджи умеренно (1-2 на сообщение) для создания дружелюбной атмосферы
- Используй сокращения естественно ("ок", "давай", "круто"), но не злоупотребляй
- Обращайся на "ты"
- Будь конкретным и полезным в ответах

==========================================================================================================
ЦЕЛЬ КАЖДОГО ДИАЛОГА
==========================================================================================================

Твоя главная цель - помочь клиенту найти подходящие товары из ассортимента мясной продукции.
Ты должен:
1. Понять потребности клиента
2. Найти подходящие товары используя доступные инструменты
3. Предоставить полную информацию о товарах (включая финальные цены)
4. При необходимости отправить фотографии товаров
5. Помочь клиенту сделать выбор

==========================================================================================================
КРИТИЧЕСКИ ВАЖНО: ПРАВИЛА ВЫБОРА ИНСТРУМЕНТОВ
==========================================================================================================

У тебя есть 5 инструментов. Ты ДОЛЖЕН самостоятельно выбирать правильный инструмент на основе запроса клиента:

1. vector_search (enhance_user_product_query) - ИСПОЛЬЗУЙ ДЛЯ ТЕКСТОВЫХ КРИТЕРИЕВ
   Когда использовать:
   - Запрос содержит текстовые критерии (тип мяса, часть туши, поставщик, регион)
   - Пользователь спрашивает "Что у вас есть?", "Покажи мясо", "Какие стейки?"
   - Запрос про товары от конкретного поставщика или региона
   - Запрос НЕ содержит числовых условий (цена, вес, скидка с числами)

   НЕ используй если:
   - Запрос содержит ЧИСЛОВЫЕ условия (цена меньше X, вес больше Y, скидка больше Z%)

2. generate_sql_from_text + execute_sql_request (text_to_sql_products) - ИСПОЛЬЗУЙ ДЛЯ ЧИСЛОВЫХ УСЛОВИЙ
   Когда использовать:
   - Запрос содержит ЧИСЛОВЫЕ условия про ЦЕНУ ("цена меньше 80", "дешевле 100 рублей")
   - Запрос содержит ЧИСЛОВЫЕ условия про ВЕС ("вес больше 5 кг", "минимальный заказ меньше 10")
   - Запрос содержит ЧИСЛОВЫЕ условия про СКИДКУ ("скидка больше 15%", "скидка от 10 до 20")
   - Запрос содержит КОМБИНАЦИЮ числовых условий

   ВАЖНО: Сначала используй generate_sql_from_text, затем execute_sql_request!

   НЕ используй если:
   - Запрос содержит ТОЛЬКО текстовые критерии БЕЗ чисел - используй vector_search

3. show_product_photos - ИСПОЛЬЗУЙ ДЛЯ ОТПРАВКИ ФОТО
   Когда использовать:
   - Пользователь просит показать фото товаров
   - После поиска товаров, если нужно визуально представить товары
   - Когда клиент хочет увидеть как выглядят товары

   Параметры: список названий товаров и номер телефона клиента

4. get_client_profile - ИСПОЛЬЗУЙ ДЛЯ ПРОФИЛЯ КЛИЕНТА
   Когда использовать:
   - Нужна информация о клиенте для персонализации ответов
   - Нужно узнать город, бизнес-область или другие данные клиента
   - Нужно адаптировать предложения под профиль клиента

   Параметр: номер телефона клиента

5. get_random_products - FALLBACK ИНСТРУМЕНТ
   Когда использовать:
   - vector_search вернул "Товары по вашему запросу не найдены"
   - execute_sql_request вернул "Товары по указанным условиям не найдены"
   - Все остальные инструменты поиска не дали результатов
   - Нужно показать примеры товаров из ассортимента

   Это инструмент последней надежды - используй его только когда ничего не найдено!

==========================================================================================================
РАСЧЕТ ФИНАЛЬНОЙ ЦЕНЫ
==========================================================================================================

КРИТИЧЕСКИ ВАЖНО: Всегда показывай клиенту ФИНАЛЬНУЮ цену (final_price_kg), а не order_price_kg!

Правила расчета финальной цены (используются значения из SYS VARIABLES):
1. Если order_price_kg < 100: final_price_kg = order_price_kg + наценка (из SYS VARIABLES)
2. Если order_price_kg >= 100: final_price_kg = order_price_kg * коэффициент (из SYS VARIABLES) + order_price_kg

Всегда выводи final_price_kg в ответах клиенту, а не order_price_kg!
Системные переменные для расчета цен находятся в блоке SYS VARIABLES выше.

==========================================================================================================
ДОСТУПНЫЕ ИНСТРУМЕНТЫ
==========================================================================================================

У тебя есть доступ к следующим инструментам:

1. vector_search - поиск товаров по текстовым критериям (семантический поиск)
2. generate_sql_from_text - генерация SQL условий из текстового описания (для числовых условий)
3. execute_sql_request - выполнение SQL запроса для поиска товаров по числовым условиям
4. show_product_photos - отправка фотографий товаров клиенту через WhatsApp
5. get_client_profile - получение профиля клиента из базы данных
6. get_random_products - получение случайных товаров (fallback инструмент)

ВАЖНО: Всегда читай docstring каждого инструмента перед использованием, чтобы понять когда его использовать!

==========================================================================================================
ИНФОРМАЦИЯ ДЛЯ ИСПОЛЬЗОВАНИЯ
==========================================================================================================

КРИТИЧЕСКИ ВАЖНО: Используй ТОЛЬКО реальные данные из базы данных!
- НЕ придумывай товары, цены, характеристики
- НЕ используй данные, которых нет в результатах инструментов
- Если товары не найдены - честно скажи об этом и предложи использовать get_random_products
- Всегда проверяй результаты инструментов перед ответом клиенту

==========================================================================================================
МИССИЯ И ПОВЕДЕНИЕ
==========================================================================================================

Твоя миссия:
- Помочь каждому клиенту найти подходящие товары
- Предоставить полную и точную информацию о товарах
- Быть дружелюбным, но профессиональным
- Использовать правильные инструменты для каждого запроса
- Всегда показывать финальные цены, а не базовые

Поведение:
- Будь проактивным: если клиент спрашивает про товары, сразу используй подходящий инструмент
- Будь конкретным: предоставляй полную информацию о товарах (название, поставщик, регион, цена, вес, упаковка и т.д.)
- Будь полезным: если товары не найдены, предложи альтернативы или используй get_random_products
- Будь внимательным: всегда проверяй результаты инструментов и используй только реальные данные

==========================================================================================================
ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ
==========================================================================================================

Пример 1: "Покажи говядину"
→ Используй vector_search с query="говядина"

Пример 2: "Товары дешевле 100 рублей"
→ Используй generate_sql_from_text с text_conditions="цена меньше 100 рублей"
→ Затем execute_sql_request с полученными SQL условиями

Пример 3: "Покажи фото стейков"
→ Сначала vector_search с query="стейки"
→ Затем show_product_photos с названиями найденных товаров

Пример 4: "Что у вас есть от Мироторг?"
→ Используй vector_search с query="товары от Мироторг"

Пример 5: "Товары не найдены"
→ Используй get_random_products для показа примеров ассортимента

==========================================================================================================

Помни: Ты - Эдуард, дружелюбный помощник, который помогает клиентам найти подходящие товары.
Используй инструменты правильно, показывай финальные цены, будь полезным и дружелюбным!"""

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
                vector_search,
                show_product_photos,
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
        self._callbacks = callbacks
        self.SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT
        self._executor_cache: dict[str, AgentExecutor] = {}
        self._cached_prompt_hash: Optional[str] = None

    def _get_prompt_hash(self, system_prompt: str) -> str:
        """Вычисляет хеш промпта для кэширования.

        Args:
            system_prompt: Системный промпт

        Returns:
            SHA256 хеш промпта
        """
        return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()

    def _create_agent_executor(self) -> AgentExecutor:
        """Создаёт AgentExecutor с промптом и инструментами.

        Примечание: callbacks не передаются при создании, они передаются в ainvoke()
        для возможности переиспользования executor.

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
            logger.info(f"[ProductAgent._get_agent_executor] Создание нового AgentExecutor (промпт изменился или кэш пуст)")

            if current_prompt_hash != self._cached_prompt_hash:
                self._executor_cache.clear()

            executor = self._create_agent_executor()
            self._executor_cache[current_prompt_hash] = executor
            self._cached_prompt_hash = current_prompt_hash

            logger.info(f"[ProductAgent._get_agent_executor] AgentExecutor закэширован (хеш: {current_prompt_hash[:8]}...)")
        else:
            logger.debug(f"[ProductAgent._get_agent_executor] Использование закэшированного AgentExecutor (хеш: {current_prompt_hash[:8]}...)")

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

        from datetime import date

        trace_name = endpoint_name or "AgentExecutor"
        langfuse_handler = LangfuseHandler(
            client_phone=client_phone,
            session_id=f"{client_phone}_{date.today()}",
            trace_name=trace_name,
        )


        try:
            logger.info(
                f"[ProductAgent.run] Начало обработки запроса для {client_phone}, topic: {topic}"
            )

            db_prompt = None
            if topic:
                try:
                    db_prompt = await get_prompt(topic)
                    if db_prompt:
                        logger.info(f"[ProductAgent.run] Загружен промпт из БД для topic '{topic}' (длина: {len(db_prompt)} символов)")
                    else:
                        logger.info(f"[ProductAgent.run] Промпт для topic '{topic}' не найден в БД, используем дефолтный")
                except Exception as e:
                    logger.warning(
                        f"[ProductAgent.run] Не удалось загрузить промпт для topic '{topic}': {e}"
                    )

            system_vars = {}
            try:
                system_vars = await get_all_system_values()
                if system_vars:
                    logger.info(f"[ProductAgent.run] Загружено системных переменных: {len(system_vars)}")
            except Exception as e:
                logger.warning(f"[ProductAgent.run] Не удалось загрузить системные переменные: {e}")


            profile_context = ""

            base_prompt = self.DEFAULT_SYSTEM_PROMPT

            if db_prompt:
                base_prompt = f"{self.DEFAULT_SYSTEM_PROMPT}\n\n==========================================================================================================\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ДЛЯ ТЕКУЩЕГО ДИАЛОГА\n==========================================================================================================\n\n{db_prompt}"
                logger.info(f"[ProductAgent.run] Добавлен промпт из БД для topic '{topic}' к системе Эдуарда")

            final_prompt = build_prompt_with_context(
                base_prompt=base_prompt,
                client_info=profile_context if profile_context else None,
                system_vars=system_vars if system_vars else None,
            )

            logger.info(f"[ProductAgent.run] Сформирован финальный промпт (длина: {len(final_prompt)} символов)")

            self.SYSTEM_PROMPT = final_prompt

            chat_history: List[BaseMessage] = []
            if self.memory is not None:
                try:
                    memory_vars = await self.memory.load_memory_variables(
                        {}, return_messages=True
                    )
                    chat_history = memory_vars.get("history", [])
                    logger.info(f"[ProductAgent.run] Загружена история диалога: {len(chat_history)} сообщений")
                except Exception as e:
                    logger.warning(f"[ProductAgent.run] Не удалось загрузить память: {e}")
                    chat_history = []
            else:
                logger.info(f"[ProductAgent.run] Память не настроена, история диалога пуста")

            input_with_context = user_input
            logger.info(f"[ProductAgent.run] User input: {user_input[:100]}...")

            try:
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

                logger.info(f"[ProductAgent.run] Подготовлено {len(callbacks_list)} callbacks для передачи в invoke()")

                logger.info(f"[ProductAgent.run] Получение AgentExecutor с промптом (длина: {len(self.SYSTEM_PROMPT)} символов)")
                agent_executor = self._get_agent_executor()
                logger.info(f"[ProductAgent.run] AgentExecutor получен, запуск агента...")

                config: RunnableConfig = {
                    "metadata": {
                        "phone": client_phone,
                        "user_id": client_phone,
                        "trace_name": trace_name,
                        "session_id": langfuse_handler.session_id,
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
                    callbacks=callbacks_list,
                )
                logger.info(f"[ProductAgent.run] AgentExecutor завершил выполнение успешно")
            except Exception as e:
                error_msg = f"Ошибка при выполнении агента: {str(e)}"
                logger.error(f"[ProductAgent.run] Ошибка AgentExecutor: {error_msg}", exc_info=True)

                try:
                    logger.info(f"[ProductAgent.run] Пытаемся использовать fallback: get_random_products")
                    fallback_result = await get_random_products.ainvoke({"limit": 2})
                    if fallback_result and "Найдено товаров" in fallback_result:
                        response_text = f"К сожалению, произошла ошибка при поиске товаров 😔. Но вот несколько товаров из нашего ассортимента:\n\n{fallback_result}\n\nПопробуйте переформулировать запрос, и я обязательно помогу найти то, что вам нужно!"
                        logger.info(f"[ProductAgent.run] Fallback успешно использован, получено товаров из get_random_products")
                    else:
                        raise Exception("Fallback не вернул товары")
                except Exception as fallback_error:
                    logger.error(f"[ProductAgent.run] Ошибка при использовании fallback: {fallback_error}", exc_info=True)
                    response_text = "Упс, что-то пошло не так 😅. Попробуйте переформулировать запрос, и я обязательно помогу!"

                return response_text

            response_text = result.get("output", "")
            logger.info(f"[ProductAgent.run] Получен ответ от агента (длина: {len(response_text)} символов)")
            if not response_text:
                logger.warning(f"[ProductAgent.run] Ответ от агента пустой, пытаемся использовать fallback")
                try:
                    fallback_result = await get_random_products.ainvoke({"limit": 2})
                    if fallback_result and "Найдено товаров" in fallback_result:
                        response_text = f"К сожалению, не удалось найти товары по вашему запросу 😔. Но вот несколько товаров из нашего ассортимента:\n\n{fallback_result}\n\nПопробуйте переформулировать запрос, и я обязательно помогу найти то, что вам нужно!"
                        logger.info(f"[ProductAgent.run] Fallback успешно использован для пустого ответа")
                    else:
                        response_text = "Упс, что-то пошло не так 😅. Попробуйте переформулировать запрос, и я обязательно помогу!"
                except Exception as fallback_error:
                    logger.error(f"[ProductAgent.run] Ошибка при использовании fallback для пустого ответа: {fallback_error}", exc_info=True)
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
                        logger.info(f"[ProductAgent.run] Сообщения сохранены в память")
                except Exception as e:
                    logger.warning(f"[ProductAgent.run] Не удалось сохранить в память: {e}")

            langfuse_handler.save_conversation_to_langfuse()
            logger.info(f"[ProductAgent.run] Данные сохранены в Langfuse")

            tools_list = sorted(list(langfuse_handler.used_tools))
            if tools_list:
                tool_type_map = {
                    "vector_search": "VECTOR SEARCH",
                    "generate_sql_from_text": "SQL GENERATOR",
                    "execute_sql_request": "SQL EXECUTOR",
                    "show_product_photos": "PHOTO SENDER",
                    "get_client_profile": "CLIENT PROFILE",
                    "get_random_products": "RANDOM PRODUCTS",
                }
                tools_summary = []
                for tool_name in tools_list:
                    tool_calls_for_tool = [
                        tc for tc in langfuse_handler.tool_calls
                        if tc.get("tool_name") == tool_name
                    ]
                    call_count = len(tool_calls_for_tool)
                    tool_type = tool_type_map.get(tool_name, "TOOL")

                    durations = [tc.get("duration") for tc in tool_calls_for_tool if tc.get("duration")]
                    duration_info = f", средняя длительность: {sum(durations)/len(durations):.2f}s" if durations else ""

                    tools_summary.append(f"{tool_type} {tool_name}({call_count}x{duration_info})")

                trace_id_info = f" (trace_id: {langfuse_handler._trace_id})" if langfuse_handler._trace_id else ""
                logger.info(
                    f"[ProductAgent.run] Использовано {len(tools_list)} инструментов для {client_phone}{trace_id_info}: {', '.join(tools_summary)}"
                )
            else:
                logger.info(f"[ProductAgent.run] Инструменты не использовались для {client_phone}")

            logger.info(f"[ProductAgent.run] Успешное завершение обработки запроса для {client_phone}")
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
