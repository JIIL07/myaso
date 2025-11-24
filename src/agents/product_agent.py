"""ProductAgent - агент для работы с продуктами и каталогом.

Использует LangChain create_agent для обработки запросов пользователей
с использованием tools для поиска товаров через семантический поиск и SQL фильтрацию.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import date
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from src.agents.middleware.tool_error_middleware import handle_tool_errors
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.config.constants import (
    AGENT_RECURSION_LIMIT,
    DEFAULT_TEMPERATURE,
    MAX_AGENT_EXECUTION_TIME,
    MAX_AGENT_ITERATIONS,
)
from src.config.settings import settings
from src.database.queries.clients_queries import get_client_is_friend
from src.utils.callbacks.langfuse_callback import LangfuseHandler
from src.utils.prompts import (
    build_prompt_with_context,
    get_all_system_values,
    get_prompt,
)

from .base_agent import BaseAgent
from .tools.client_tools import get_client_profile
from .tools.context_tools import (
    get_conversation_context,
    save_product_ids_to_context,
    set_photo_requirement,
)
from .tools.media_tools import show_product_photos
from .tools.product_tools import get_random_products, vector_search
from .tools.sql_tools import create_sql_tools
from .tools.context_vars import client_phone_context

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

    Использует LangChain create_agent с tools для поиска товаров через:
    - vector_search для семантического поиска
    - generate_sql_from_text + execute_sql_query для фильтрации по параметрам
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
                # Не логируем инициализацию LLM - это не важно
            except Exception as e:
                logger.error(
                    f"[ProductAgent] Ошибка инициализации LLM: {e}",
                    exc_info=True
                )
                raise ValueError(f"Не удалось инициализировать LLM: {e}") from e

        if tools is None:
            # Базовые инструменты (все статические, используют contextvars для client_phone)
            tools = [
                get_client_profile,
                vector_search,
                get_random_products,
            ]

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT
        self._agent_cache: dict[str, Any] = {}
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
        """Собирает промпт для модели (требуется BaseAgent, но не используется)."""
        return user_input

    def _create_tools(self) -> List[Any]:
        """Создаёт и возвращает список инструментов (требуется BaseAgent, но не используется)."""
        return self.tools

    def _create_agent(
        self, tools: Optional[List[Any]] = None
    ) -> Any:
        """Создаёт агента через create_agent API.

        Args:
            tools: Список инструментов (если None, используются self.tools)

        Returns:
            Runnable объект агента
        """
        system_prompt = self.SYSTEM_PROMPT
        agent_tools = tools or self.tools

        # Создаем middleware для ограничения вызовов модели и обработки ошибок
        middleware = [handle_tool_errors]
        if MAX_AGENT_ITERATIONS > 0:
            middleware.append(
                ModelCallLimitMiddleware(
                    run_limit=MAX_AGENT_ITERATIONS,
                    exit_behavior="end",
                )
            )

        agent = create_agent(
            model=self.llm,
            tools=agent_tools,
            system_prompt=system_prompt,
            middleware=middleware if middleware else None,
        )

        return agent

    def _get_agent(
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
            agent = self._create_agent(tools=agent_tools)
            self._agent_cache[cache_key] = agent
            self._cached_prompt_hash = current_prompt_hash
        else:
            # Проверяем, изменился ли промпт
            if current_prompt_hash != self._cached_prompt_hash:
                # Промпт изменился - очищаем кэш и создаем новый агент
                self._agent_cache.clear()
                agent = self._create_agent(tools=agent_tools)
                self._agent_cache[cache_key] = agent
                self._cached_prompt_hash = current_prompt_hash
            else:
                agent = self._agent_cache[cache_key]

        return agent

    async def _load_prompt_and_context(
        self,
        topic: Optional[str],
        client_phone: str,
    ) -> tuple[str, Dict[str, str], str, List[BaseMessage]]:
        """Загружает промпт, системные переменные, информацию о клиенте и историю.

        Args:
            topic: Тема диалога для загрузки промпта из БД
            client_phone: Номер телефона клиента

        Returns:
            Кортеж (base_prompt, system_vars, client_info, chat_history)
        """
        db_prompt = None
        if topic:
            try:
                db_prompt = await get_prompt(topic)
                if not db_prompt:
                    logger.warning(f"[ProductAgent] Промпт для topic '{topic}' не найден в БД")
            except Exception as e:
                logger.error(f"[ProductAgent] Не удалось загрузить промпт для topic '{topic}': {e}")

        system_vars = {}
        try:
            system_vars = await get_all_system_values()
        except Exception as e:
            logger.error(f"[ProductAgent] Не удалось загрузить системные переменные: {e}")

        if db_prompt:
            base_prompt = db_prompt + f"\n\n{self.DEFAULT_SYSTEM_PROMPT}"
        else:
            base_prompt = self.DEFAULT_SYSTEM_PROMPT

        chat_history: List[BaseMessage] = []
        if self.memory is not None:
            try:
                if not hasattr(self.memory, 'async_initialized') or not self.memory.async_initialized:
                    logger.warning(f"[ProductAgent] Память не инициализирована для {client_phone}, пропускаем загрузку истории")
                    chat_history = []
                else:
                    memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
                    chat_history = memory_vars.get("history", [])
            except Exception as e:
                logger.error(f"[ProductAgent] Не удалось загрузить память: {e}", exc_info=True)
                chat_history = []

        client_is_friend = False
        try:
            client_is_friend = await get_client_is_friend(client_phone)
        except Exception as e:
            logger.error(f"[ProductAgent] Не удалось получить статус дружбы клиента: {e}", exc_info=True)

        client_info_parts = [
            f"Номер телефона: {client_phone}",
            f"Статус дружбы (it_is_friend): {client_is_friend}",
        ]
        if client_is_friend:
            client_info_parts.append("ОБРАЩЕНИЕ: Используй 'ты' (неформальное общение)")
        else:
            client_info_parts.append("ОБРАЩЕНИЕ: Используй 'вы' (формальное общение)")

        client_info = "\n".join(client_info_parts)

        return base_prompt, system_vars, client_info, chat_history

    def _is_first_message(self, chat_history: List[BaseMessage]) -> bool:
        """Определяет, является ли это первым сообщением в разговоре.

        Args:
            chat_history: История сообщений

        Returns:
            True если это первое сообщение (нет HumanMessage в истории), False иначе
        """
        # Проверяем наличие HumanMessage в истории
        # Если нет HumanMessage - это первое сообщение
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                return False
        return True

    def _prepare_messages(
        self,
        user_input: str,
        chat_history: List[BaseMessage],
    ) -> str:
        """Подготавливает сообщения с контекстом для агента.

        Args:
            user_input: Текст запроса пользователя
            chat_history: История сообщений

        Returns:
            Текст запроса с добавленным контекстом
        """
        is_second_message = False
        if len(chat_history) == 1:
            if isinstance(chat_history[0], AIMessage):
                is_second_message = True
        elif len(chat_history) == 2:
            if isinstance(chat_history[0], AIMessage) and isinstance(chat_history[1], HumanMessage):
                is_second_message = True

        client_greeted = is_greeting_message(user_input)

        context_parts = []
        if client_greeted:
            if is_second_message:
                context_parts.append("ВАЖНО: Это второе сообщение, но клиент поздоровался с тобой. Поздоровайся в ответ, затем продолжай общение.")
            else:
                context_parts.append("ВАЖНО: Клиент поздоровался с тобой. Поздоровайся в ответ, затем продолжай общение.")
        elif is_second_message:
            context_parts.append("ВАЖНО: Это второе сообщение в разговоре. НЕ используй приветствие, сразу переходи к делу.")

        if context_parts:
            return user_input + "\n\n" + "\n".join(context_parts)
        return user_input

    async def _execute_agent(
        self,
        messages: List[BaseMessage],
        agent_tools: List[Any],
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        """Выполняет агента с заданными сообщениями и инструментами.

        Args:
            messages: Список сообщений для агента
            agent_tools: Список инструментов агента
            config: Конфигурация для выполнения

        Returns:
            Результат выполнения агента

        Raises:
            Exception: Если произошла ошибка при выполнении агента
        """
        agent = self._get_agent(tools=agent_tools)

        if MAX_AGENT_EXECUTION_TIME > 0:
            result = await asyncio.wait_for(
                agent.ainvoke({"messages": messages}, config=config),
                timeout=MAX_AGENT_EXECUTION_TIME,
            )
        else:
            result = await agent.ainvoke({"messages": messages}, config=config)

        return result

    def _extract_response(self, result: Dict[str, Any]) -> str:
        """Извлекает ответ агента из результата выполнения.

        Args:
            result: Результат выполнения агента

        Returns:
            Текст ответа агента
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
                    response_text = " ".join(text_parts) if text_parts else str(content)
                else:
                    response_text = str(content) if content else ""
                break

        if not response_text:
            response_text = result.get("output", "")
        if not response_text:
            response_text = "Упс, что-то пошло не так 😅. Попробуйте переформулировать запрос, и я обязательно помогу!"

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
            if not hasattr(self.memory, 'async_initialized') or not self.memory.async_initialized:
                logger.warning(f"[ProductAgent] Память не инициализирована для {client_phone}, пропускаем сохранение")
                return

            # Не сохраняем user_input если это первое сообщение
            is_first = self._is_first_message(chat_history)
            if not is_first:
                await self.memory.add_messages([HumanMessage(content=user_input)])
            await self.memory.add_messages([AIMessage(content=response_text)])
        except Exception as e:
            logger.error(f"[ProductAgent] Не удалось сохранить в память для {client_phone}: {e}", exc_info=True)

    async def run(
        self,
        user_input: str,
        client_phone: str,
        topic: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> str:
        """Запускает агента для обработки запроса пользователя.

        Использует современный LangChain API create_agent для выполнения агента.
        Формат вызова: {"messages": [...]} вместо старого {"input": ..., "chat_history": ...}.
        Результат: {"messages": [...]} с последним AIMessage в качестве ответа.

        Args:
            user_input: Текст запроса пользователя
            client_phone: Номер телефона клиента
            topic: Тема диалога для загрузки промпта из БД (опционально)
            endpoint_name: Имя endpoint для трейсинга

        Returns:
            Строка с ответом агента (извлекается из последнего AIMessage)
        """
        trace_name = endpoint_name or "ProductAgent"

        langfuse_handler = LangfuseHandler(
            client_phone=client_phone,
            session_id=f"{client_phone}_{date.today()}",
            trace_name=trace_name,
        )

        try:
            logger.info(f"[ProductAgent.run] Начало обработки для {client_phone}, topic: {topic}")

            base_prompt, system_vars, client_info, chat_history = await self._load_prompt_and_context(
                topic, client_phone
            )

            # Загружаем дополнительные промпты для инструментов
            tool_prompts = []
            for prompt_topic in [
                "SQL Generation Rules",
                "Vector Search Instructions",
                "Photo Sending Instructions",
                "Tool Usage Guidelines",
            ]:
                tool_prompt = await get_prompt(prompt_topic)
                if tool_prompt:
                    tool_prompts.append(tool_prompt)

            # Объединяем все промпты
            if tool_prompts:
                tools_section = "\n\n".join([f"--- {topic} ---\n{p}" for topic, p in zip(
                    ["SQL Generation Rules", "Vector Search Instructions", "Photo Sending Instructions", "Tool Usage Guidelines"],
                    tool_prompts
                ) if p])
                base_prompt = f"{base_prompt}\n\n{tools_section}"

            final_prompt = build_prompt_with_context(
                base_prompt=base_prompt,
                client_info=client_info,
                system_vars=system_vars if system_vars else None,
            )
            self.SYSTEM_PROMPT = final_prompt

            input_with_context = self._prepare_messages(user_input, chat_history)

            # Загружаем контекст агента (будет загружен из БД или кэша)
            # Контекст автоматически сохраняется через инструменты при изменениях
            from .tools.context_tools import get_agent_context_async
            await get_agent_context_async(client_phone)  # Загружаем в кэш
            
            # Определяем, является ли это первым сообщением
            is_first = self._is_first_message(chat_history)
            
            # Устанавливаем client_phone в контекст для статических инструментов
            client_phone_context.set(client_phone)
            
            # Статические инструменты (используют contextvars для получения client_phone)
            context_tools = [set_photo_requirement, get_conversation_context]
            sql_tools = create_sql_tools()
            media_tools = [show_product_photos]
            
            # Все инструменты теперь статические
            agent_tools = self.tools + sql_tools + media_tools + context_tools

            callbacks_list = [
                langfuse_handler,
                StdOutCallbackHandler(),
            ]

            config: RunnableConfig = {
                "callbacks": callbacks_list,
                "metadata": {
                    "phone": client_phone,
                    "user_id": client_phone,
                    "trace_name": trace_name,
                },
                "run_name": trace_name,
                "tags": ["product_agent", "conversation", trace_name],
                # Устанавливаем лимит рекурсии для LangGraph
                # Это предотвращает ошибку "Recursion limit reached"
                "recursion_limit": AGENT_RECURSION_LIMIT,
            }

            messages = []
            if chat_history:
                messages.extend(chat_history)
            messages.append(HumanMessage(content=input_with_context))

            try:
                result = await self._execute_agent(messages, agent_tools, config)
            except asyncio.TimeoutError:
                error_msg = f"Агент превысил максимальное время выполнения ({MAX_AGENT_EXECUTION_TIME} секунд)"
                logger.error(f"[ProductAgent.run] Timeout агента: {error_msg}")
                raise Exception(error_msg)
            except Exception as e:
                error_msg = f"Ошибка при выполнении агента: {str(e)}"
                logger.error(f"[ProductAgent.run] Ошибка агента: {error_msg}", exc_info=True)
                raise Exception(error_msg) from e
            finally:
                # Очищаем контекст после выполнения
                client_phone_context.set('')

            response_text = self._extract_response(result)

            messages_result = result.get("messages", [])
            if messages_result:
                tool_messages = [msg for msg in messages_result if isinstance(msg, ToolMessage)]
                steps_count = len(tool_messages) if tool_messages else 0
                
                # Обрабатываем artifacts из ToolMessage и сохраняем product_ids в context7
                for tool_msg in tool_messages:
                    # Проверяем наличие artifact в ToolMessage
                    if hasattr(tool_msg, 'artifact') and tool_msg.artifact is not None:
                        # Проверяем, что это список ID товаров
                        if isinstance(tool_msg.artifact, list) and len(tool_msg.artifact) > 0:
                            # Проверяем, что все элементы - числа
                            if all(isinstance(x, int) for x in tool_msg.artifact):
                                try:
                                    await save_product_ids_to_context(client_phone, tool_msg.artifact)
                                    tool_name = getattr(tool_msg, 'name', 'unknown')
                                    logger.info(
                                        f"[ProductAgent.run] Сохранено {len(tool_msg.artifact)} product_ids "
                                        f"в context7 для {client_phone} из инструмента {tool_name}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"[ProductAgent.run] Ошибка сохранения product_ids в context7: {e}",
                                        exc_info=True
                                    )

                if steps_count == 0:
                    logger.warning(
                        f"[ProductAgent.run] ⚠️ НЕТ ВЫЗОВОВ ИНСТРУМЕНТОВ для {client_phone}: "
                        f"агент ответил БЕЗ вызова инструментов! "
                        f"user_input='{user_input[:200]}'"
                    )
                else:
                    logger.info(
                        f"[ProductAgent.run] ✅ Запрос обработан для {client_phone}: "
                        f"использовано {steps_count} инструмент(ов), "
                        f"response_length={len(response_text)}"
                    )

            await self._save_to_memory(user_input, response_text, chat_history, client_phone)

            langfuse_handler.save_conversation_to_langfuse()

            return response_text

        except Exception as e:
            error_msg = "Ой, что-то пошло не так 😔. Попробуйте написать еще раз, пожалуйста!"
            logger.error(f"[ProductAgent.run] Ошибка ProductAgent: {str(e)}", exc_info=True)

            try:
                langfuse_handler.save_conversation_to_langfuse()
            except Exception as langfuse_error:
                logger.warning(f"Не удалось сохранить ошибку в LangFuse: {langfuse_error}")

            return error_msg