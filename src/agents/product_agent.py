"""ProductAgent - агент для работы с продуктами и каталогом.

Использует LangChain AgentExecutor для обработки запросов пользователей
с использованием tools для поиска товаров через семантический поиск и SQL фильтрацию.
"""

from __future__ import annotations

from typing import Any, List, Optional
import logging
import hashlib
import json
from datetime import date
from langchain_classic.agents import (
    AgentExecutor,
    create_openai_tools_agent,
    create_react_agent,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
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
from .tools.product_tools import vector_search, get_random_products
from .tools.sql_tools import execute_sql_request, create_sql_tools
from .tools.client_tools import get_client_profile
from .tools.media_tools import create_media_tools
from src.utils.prompts import (
    get_prompt,
    get_all_system_values,
    build_prompt_with_context,
)
from src.database.queries.clients_queries import get_client_is_friend

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
    - t + execute_sql_request для фильтрации по параметрам
    - get_random_products как fallback
    """

    DEFAULT_SYSTEM_PROMPT = """

==========================================================================================================
ДОСТУПНЫЕ ИНСТРУМЕНТЫ (TOOLS)
==========================================================================================================

У тебя есть следующие инструменты для работы:

1. vector_search(query: str) → str

  НАЗНАЧЕНИЕ: Семантический поиск товаров по текстовому запросу (векторный поиск)

  ИСПОЛЬЗУЙ ДЛЯ:
  - Текстовые запросы: "что у вас есть?", "покажи мясо", "какие стейки?"
  - Поиск по типу/части: "грудинка свиная", "говядина", "стейки", "полуфабрикаты"
  - Поиск по поставщику: "товары от Коралл", "продукция Мироторг"
  - Поиск по региону: "мясо из Сибири", "товары из Бурятии"
  - Комбинации текстовых критериев: "свинина охлажденная", "стейки от Коралл"

  НЕ ИСПОЛЬЗУЙ ДЛЯ:
  - Числовые условия: "цена меньше 100" → используй generate_sql_from_text
  - Условия по весу: "вес больше 5 кг" → используй generate_sql_from_text
  - Условия по скидке: "скидка больше 15%" → используй generate_sql_from_text

  ВОЗВРАЩАЕТ: Список найденных товаров (до 50) с ID в секции [PRODUCT_IDS]

2. generate_sql_from_text(text_conditions: str, topic: Optional[str] = None) → str

  НАЗНАЧЕНИЕ: Генерирует SQL WHERE условия из текстового описания на русском языке

  ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ДЛЯ:
  - Числовые условия по ЦЕНЕ: "цена меньше 80", "дешевле 100 рублей", "цена от 50 до 200"
  - Числовые условия по ВЕСУ: "вес больше 5 кг", "минимальный заказ меньше 10"
  - Числовые условия по СКИДКЕ: "скидка больше 15%", "скидка от 10 до 20"
  - Комбинации числовых условий: "цена меньше 100 и скидка больше 10%"
  - Пустые запросы или init_conversation → передай описание темы/категории

  НЕ ИСПОЛЬЗУЙ ДЛЯ:
  - Только название поставщика БЕЗ чисел: "товары от Мироторг" → используй vector_search
  - Только название региона БЕЗ чисел: "мясо из Сибири" → используй vector_search
  - Только текстовые критерии БЕЗ чисел: "говядина", "стейки" → используй vector_search

  ВАЖНО: После вызова generate_sql_from_text ОБЯЗАТЕЛЬНО вызови execute_sql_request с полученными SQL условиями!

  ВОЗВРАЩАЕТ: SQL WHERE условия (без ключевого слова WHERE) для использования в execute_sql_request

3. execute_sql_request(sql_conditions: str, limit: int = 50) → str

  НАЗНАЧЕНИЕ: Выполняет SQL запрос с WHERE условиями и возвращает товары

  ИСПОЛЬЗУЙ КОГДА:
   - У тебя есть готовые SQL WHERE условия от generate_sql_from_text
   - Нужно выполнить SQL запрос для поиска товаров по числовым условиям

  НЕ ИСПОЛЬЗУЙ ЕСЛИ:
   - У тебя нет готовых SQL условий → сначала используй generate_sql_from_text
   - Запрос не содержит числовых условий → используй vector_search

  ВАЖНО: Всегда используй в паре с generate_sql_from_text:
   1. generate_sql_from_text("цена меньше 100") → получаешь SQL условия
   2. execute_sql_request(sql_conditions) → получаешь товары

  ВОЗВРАЩАЕТ: Список найденных товаров (до 50) с ID в секции [PRODUCT_IDS]

4. get_random_products(limit: int = 10) → str

  НАЗНАЧЕНИЕ: Получает случайные товары из ассортимента (FALLBACK инструмент)

  ИСПОЛЬЗУЙ ТОЛЬКО КОГДА:
  - vector_search вернул "Товары по вашему запросу не найдены"
  - execute_sql_request вернул "Товары по указанным условиям не найдены"
  - Все остальные инструменты поиска не дали результатов
  - Нужно показать примеры товаров из ассортимента когда ничего не найдено

  НЕ ИСПОЛЬЗУЙ ЕСЛИ:
  - vector_search или execute_sql_request уже нашли товары
  - Есть конкретный запрос, который можно обработать другими инструментами

  ВАЖНО: Это инструмент последней надежды! Всегда сначала пробуй vector_search или generate_sql_from_text + execute_sql_request.

  ВОЗВРАЩАЕТ: Список случайных товаров (до 20) с ID в секции [PRODUCT_IDS]

5. show_product_photos(product_ids: List[int]) → str

  НАЗНАЧЕНИЕ: Отправляет фотографии товаров клиенту через WhatsApp

  ИСПОЛЬЗУЙ ТОЛЬКО В ДВУХ СЛУЧАЯХ:

  1. Когда пользователь ЯВНО просит показать/отправить фото:
    - "отправь фото", "покажи фото", "фотографии", "покажи фотографии товаров"
    - "отправь фото этих товаров", "хочу увидеть фото"
    - "покажи фото грудинки свиной", "отправь фото товаров от Коралл"

  2. При инициализации разговора (init_conversation) - первое сообщение в диалоге

  ВАЖНО - ДВА СЦЕНАРИЯ:

   СЦЕНАРИЙ 1: Клиент просит фото КОНКРЕТНЫХ товаров
   Пример: "покажи фото грудинки свиной", "отправь фото товаров от Коралл"
   Алгоритм:
   1. СНАЧАЛА найди товары: vector_search("грудинка свиная") или execute_sql_request
   2. Получи ID из ответа: [PRODUCT_IDS]{"product_ids": [789, 790]}[/PRODUCT_IDS]
   3. ПОТОМ отправь фото: show_product_photos product_ids=[789, 790]

   СЦЕНАРИЙ 2: Клиент просто просит "отправь фото" (без уточнения)
   Пример: После поиска "есть коралл" → клиент: "отправь фото"
   Алгоритм:
   1. Используй ID из ПОСЛЕДНЕГО ответа инструментов поиска в chat_history
   2. Извлеки product_ids из секции [PRODUCT_IDS] из последнего ответа
   3. show_product_photos product_ids=[извлеченные ID]

  ВОЗВРАЩАЕТ: Статус отправки фотографий (количество отправленных, не отправленных, не найденных товаров)

6. get_client_profile(phone: str) → str

   НАЗНАЧЕНИЕ: Получает профиль клиента из базы данных

  ИСПОЛЬЗУЙ КОГДА:
  - Нужна информация о клиенте для персонализации ответов
  - Нужно узнать город клиента для предложения товаров из его региона
  - Нужно узнать бизнес-область клиента для адаптации предложений
  - Нужно адаптировать ответы под профиль клиента

  НЕ ИСПОЛЬЗУЙ ЕСЛИ:
  - Информация о клиенте не нужна для ответа
  - Запрос не требует персонализации

  ВОЗВРАЩАЕТ: Информация о профиле клиента (имя, контакты, город, бизнес-данные, предпочтения)

==========================================================================================================
ПРАВИЛА ИСПОЛЬЗОВАНИЯ ИНСТРУМЕНТОВ
==========================================================================================================

АЛГОРИТМ ВЫБОРА ИНСТРУМЕНТА:

1. Если запрос содержит ЧИСЛОВЫЕ условия (цена, вес, скидка):
   → generate_sql_from_text → execute_sql_request

2. Если запрос содержит ТОЛЬКО текстовые критерии (название, поставщик, регион):
   → vector_search

3. Если все инструменты поиска не дали результатов:
   → get_random_products (fallback)

4. Если клиент просит фото:
   → Сначала найди товары (vector_search или execute_sql_request), затем show_product_photos

5. Если нужна информация о клиенте:
   → get_client_profile или get_client_orders

==========================================================================================================
ИСПОЛЬЗОВАНИЕ ИНСТРУМЕНТОВ ПОИСКА С ПАРАМЕТРОМ require_photo
==========================================================================================================

ВАЖНО: Когда клиент запрашивает ФОТО товаров, ты ДОЛЖЕН использовать параметр require_photo=True!

ПРАВИЛА ИСПОЛЬЗОВАНИЯ require_photo=True:
1. Используй require_photo=True в vector_search, execute_sql_request, get_random_products когда:
   - Клиент говорит "отправь фото", "покажи фото", "фото грудинки", "отправь фото товаров"
   - Клиент просит увидеть фотографии товаров
   - В запросе есть слова "фото", "фотографи" в контексте просьбы показать

2. Примеры правильного использования:
   - Запрос: "отправь фото грудинки свиной" → vector_search(query="грудинка свиная", require_photo=True)
   - Запрос: "покажи фото товаров от Коралл" → execute_sql_request(sql_conditions="supplier_name ILIKE '%коралл%'", require_photo=True)
   - Запрос: "хочу увидеть фото стейков" → vector_search(query="стейки", require_photo=True)

3. Когда require_photo=True:
   - Инструменты возвращают ТОЛЬКО товары с фотографиями
   - Если товаров с фото не найдено, инструмент вернет сообщение об этом
   - После поиска с require_photo=True, обязательно вызови show_product_photos для отправки фото

4. Когда require_photo=False (по умолчанию):
   - Инструменты возвращают все товары независимо от наличия фото
   - Используй когда клиент просто спрашивает о товарах без запроса на фото

==========================================================================================================
Ограничения
==========================================================================================================

- Информацию, которая тебе доступна (цены, товары, регион происхождения, регион клиента и т.д.)
- Твои инструменты поиска: vector_search, generate_sql_from_text, execute_sql_request, get_random_products
- Инструменты для работы с клиентом: get_client_profile
- Инструмент для отправки фото: show_product_photos
- Данные из результатов поиска товаров
- Системные переменные из блока SYS VARIABLES для расчета цен
"""

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

        Args:
            system_prompt: Системный промпт

        Returns:
            Хеш промпта
        """
        return hashlib.sha256(system_prompt.encode('utf-8')).hexdigest()

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

    def _get_agent_executor(
        self, callbacks: Optional[List[Any]] = None, tools: Optional[List[Any]] = None
    ) -> AgentExecutor:
        """Получает AgentExecutor из кэша или создает новый.

        Кэширует AgentExecutor по хешу текущего SYSTEM_PROMPT и инструментов.
        Если промпт или инструменты изменились, создает новый executor.
        
        ВАЖНО: Если переданы динамические инструменты (tools != None), кэширование
        происходит по комбинации промпта и инструментов.

        Args:
            callbacks: Список callbacks для AgentExecutor
            tools: Список инструментов (если None, используются self.tools)

        Returns:
            AgentExecutor для выполнения агента
        """
        current_prompt_hash = self._get_prompt_hash(self.SYSTEM_PROMPT)
        agent_tools = tools or self.tools
        
        if tools is not None:
            tools_hash = str(sorted([getattr(t, 'name', str(t)) for t in agent_tools]))
            cache_key = f"{current_prompt_hash}_{tools_hash}"
            
            if cache_key not in self._executor_cache:
                executor = self.create_agent_executor(callbacks=callbacks, tools=agent_tools)
                self._executor_cache[cache_key] = executor
            
            return self._executor_cache[cache_key]
        else:
            if current_prompt_hash != self._cached_prompt_hash or current_prompt_hash not in self._executor_cache:
                if current_prompt_hash != self._cached_prompt_hash:
                    self._executor_cache.clear()

                executor = self.create_agent_executor(callbacks=callbacks, tools=agent_tools)
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

        logger.info(
            f"[ProductAgent.run] LangfuseHandler создан для {client_phone}, "
            f"type={type(langfuse_handler).__name__}"
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
            else:
                base_prompt = self.DEFAULT_SYSTEM_PROMPT

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

                logger.info(
                    f"[ProductAgent.run] Подготовлено {len(callbacks_list)} callbacks: "
                    f"{[type(cb).__name__ for cb in callbacks_list]}"
                )

                agent_executor = self._get_agent_executor(callbacks=None, tools=agent_tools)

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
            
            reasoning_info = "AgentExecutor выполнен успешно"
            if result:
                intermediate_steps = result.get("intermediate_steps", [])
                if intermediate_steps:
                    reasoning_info = f"Выполнено {len(intermediate_steps)} шагов агента"
                logger.info(
                    f"[ProductAgent.run] Запрос обработан: "
                    f"user_input={user_input[:100]}, "
                    f"reasoning={reasoning_info}, "
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
