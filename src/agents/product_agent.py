"""ProductAgent - агент для работы с продуктами и каталогом.

Использует LangChain create_agent для обработки запросов пользователей
с использованием tools для поиска товаров через семантический поиск и SQL фильтрацию.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolRetryMiddleware,
)
from typing_extensions import NotRequired

from src.agents.middleware.tool_error_middleware import handle_tool_errors
from src.agents.middleware.product_ids_middleware import save_product_ids_middleware
from langchain_core.callbacks.stdout import StdOutCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.utils.rules import (
    get_all_instruction_rules,
    get_rule_as_float,
    get_rule_as_int,
)
from src.config.settings import settings
from src.database.queries.clients_queries import get_client_is_friend
from src.utils.callbacks.langfuse_callback import LangfuseHandler
from src.utils.memory_utils import is_memory_initialized
from src.utils.prompts import (
    build_prompt_with_context,
    get_all_system_values,
    get_prompt,
)

from .base_agent import BaseAgent
from .tools.client_tools import get_client_profile, get_client_orders, get_last_order
from .tools.context_tools import (
    get_conversation_context,
    save_product_ids_to_context,
    set_photo_requirement,
)
from .tools.media_tools import show_product_photos, send_pricelist
from .tools.price_tools import calculate_product_price
from .tools.product_tools import (
    get_random_products,
    vector_search,
    get_product_by_title,
    find_similar_products,
    compare_products,
)
from .tools.sql_tools import create_sql_tools
from .tools.context_vars import client_phone_context

logger = logging.getLogger(__name__)


@dataclass
class ProductAgentContext:
    """Контекст для ProductAgent.
    
    Содержит статическую информацию, которая не меняется во время выполнения агента.
    """
    client_phone: str


class ProductAgentState(AgentState):
    """Состояние ProductAgent.
    
    Расширяет базовое AgentState дополнительными полями для отслеживания
    состояния агента во время выполнения.
    """
    product_ids: NotRequired[List[int]]
    require_photo: NotRequired[bool]


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
                    temperature=0.5,
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
                get_client_orders,
                get_last_order,
                vector_search,
                get_product_by_title,
                find_similar_products,
                compare_products,
                get_random_products,
                calculate_product_price,
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

    async def _create_agent(
        self, tools: Optional[List[Any]] = None, max_iterations: Optional[int] = None
    ) -> Any:
        """Создаёт агента через create_agent API с валидацией конфигурации.

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
            try:
                max_iterations = await get_rule_as_int("MAX_AGENT_ITERATIONS")
            except Exception:
                max_iterations = 1000
        
        max_iterations = max(1, min(10000, max_iterations))

        # Загружаем настройки retry для инструментов из БД
        try:
            tool_retry_max_retries = await get_rule_as_int("TOOL_RETRY_MAX_RETRIES")
            tool_retry_max_retries = max(0, min(5, tool_retry_max_retries))
        except Exception:
            tool_retry_max_retries = 3
        
        try:
            tool_retry_backoff_factor = await get_rule_as_float("TOOL_RETRY_BACKOFF_FACTOR")
            tool_retry_backoff_factor = max(1.0, min(5.0, tool_retry_backoff_factor))
        except Exception:
            tool_retry_backoff_factor = 2.0
        
        try:
            tool_retry_initial_delay = await get_rule_as_float("TOOL_RETRY_INITIAL_DELAY")
            tool_retry_initial_delay = max(0.1, min(10.0, tool_retry_initial_delay))
        except Exception:
            tool_retry_initial_delay = 1.0

        try:
            model_retry_max_retries = await get_rule_as_int("MODEL_RETRY_MAX_RETRIES")
            model_retry_max_retries = max(0, min(5, model_retry_max_retries))
        except Exception:
            model_retry_max_retries = 2
        
        try:
            model_retry_backoff_factor = await get_rule_as_float("MODEL_RETRY_BACKOFF_FACTOR")
            model_retry_backoff_factor = max(1.0, min(5.0, model_retry_backoff_factor))
        except Exception:
            model_retry_backoff_factor = 2.0
        
        try:
            model_retry_initial_delay = await get_rule_as_float("MODEL_RETRY_INITIAL_DELAY")
            model_retry_initial_delay = max(0.1, min(10.0, model_retry_initial_delay))
        except Exception:
            model_retry_initial_delay = 1.0

        middleware = [
            ModelRetryMiddleware(
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
            base_prompt = f"{db_prompt}\n\n{self.DEFAULT_SYSTEM_PROMPT}".strip()
        else:
            base_prompt = self.DEFAULT_SYSTEM_PROMPT

        chat_history: List[BaseMessage] = []
        if self.memory and is_memory_initialized(self.memory):
            try:
                memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
                chat_history = memory_vars.get("history", [])
            except Exception as e:
                logger.error(f"[ProductAgent] Ошибка загрузки памяти: {e}", exc_info=True)

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

        Returns:
            Результат выполнения агента

        Raises:
            ValueError: Если входные данные невалидны
            AgentTimeoutError: Если агент превысил время выполнения
            AgentExecutionError: Если произошла ошибка при выполнении агента после всех попыток
        """
        agent = await self._get_agent(tools=agent_tools)
        
        try:
            max_execution_time = await get_rule_as_int("MAX_AGENT_EXECUTION_TIME")
            max_execution_time = max(1, min(7200, max_execution_time))
        except Exception:
            max_execution_time = 3600
            logger.warning("[ProductAgent._execute_agent] Не удалось загрузить MAX_AGENT_EXECUTION_TIME, используем 3600")
        
        # Выполняем агента с timeout
        # Retry для вызовов модели обрабатывается через ModelRetryMiddleware в middleware
        try:
            if max_execution_time > 0:
                result = await asyncio.wait_for(
                    agent.ainvoke({"messages": messages}, config=config),
                    timeout=max_execution_time,
                )
            else:
                result = await agent.ainvoke({"messages": messages}, config=config)
            return result
        except asyncio.TimeoutError as e:
            from src.utils.exceptions import AgentTimeoutError
            error_msg = f"Агент превысил максимальное время выполнения ({max_execution_time} секунд)"
            logger.error(f"[ProductAgent._execute_agent] Timeout агента: {error_msg}")
            raise AgentTimeoutError(error_msg, {"timeout": max_execution_time}) from e
        except Exception as e:
            from src.utils.exceptions import AgentExecutionError
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
        
        # Очищаем форматирование ответа (убираем лишние пробелы и переносы строк)
        import re
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r' {2,}', ' ', response)
        return response.strip()

    async def _extract_and_save_product_ids(
        self, 
        result: Dict[str, Any], 
        client_phone: str
    ) -> None:
        """Извлекает product_ids из artifacts инструментов поиска и сохраняет в контекст.
        
        Использует лучшие практики LangChain для работы с artifacts:
        - Инструменты с response_format="content_and_artifact" автоматически сохраняют
          второй элемент кортежа в ToolMessage.artifact
        - Artifact может быть списком или одиночным значением
        
        Args:
            result: Результат выполнения агента (содержит messages)
            client_phone: Номер телефона клиента для сохранения контекста
        """
        messages_result = result.get("messages", [])
        if not messages_result:
            return
        
        # Инструменты поиска товаров, которые возвращают product_ids как artifacts
        PRODUCT_SEARCH_TOOLS = {
            'vector_search',
            'execute_sql_query',
            'get_random_products',
            'find_similar_products',
            'get_product_by_title',
        }
        
        all_product_ids = []
        tool_messages = [msg for msg in messages_result if isinstance(msg, ToolMessage)]
        
        # Создаем словарь для быстрого поиска имени инструмента по tool_call_id
        tool_call_id_to_name = {}
        for msg in messages_result:
            if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_call_id = None
                    tool_name = None
                    
                    # Обрабатываем разные форматы tool_call
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get('id')
                        tool_name = tool_call.get('name')
                    elif hasattr(tool_call, 'id'):
                        tool_call_id = tool_call.id
                        tool_name = getattr(tool_call, 'name', None)
                    
                    if tool_call_id and tool_name:
                        tool_call_id_to_name[tool_call_id] = tool_name
        
        for tool_msg in tool_messages:
            # Получаем имя инструмента через tool_call_id
            tool_name = None
            tool_call_id = None
            
            if hasattr(tool_msg, 'tool_call_id'):
                tool_call_id = tool_msg.tool_call_id
                tool_name = tool_call_id_to_name.get(tool_call_id)
                
                if not tool_name:
                    logger.debug(
                        f"[ProductAgent._extract_and_save_product_ids] "
                        f"Не найдено имя инструмента для tool_call_id: {tool_call_id}"
                    )
            
            # Fallback: пытаемся получить имя из атрибутов сообщения (обычно не работает для ToolMessage)
            if not tool_name and hasattr(tool_msg, 'name'):
                tool_name = tool_msg.name
                logger.debug(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Получено имя инструмента из атрибута name: {tool_name}"
                )
            
            if not tool_name:
                logger.debug(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Не удалось определить имя инструмента для ToolMessage с tool_call_id: {tool_call_id}"
                )
                continue
            
            if tool_name not in PRODUCT_SEARCH_TOOLS:
                logger.debug(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Инструмент {tool_name} не является инструментом поиска товаров, пропускаем"
                )
                continue
            
            # Извлекаем artifact из ToolMessage
            # В LangChain artifact хранится в атрибуте artifact
            artifact = None
            if hasattr(tool_msg, 'artifact'):
                artifact = tool_msg.artifact
            elif hasattr(tool_msg, 'additional_kwargs') and 'artifact' in tool_msg.additional_kwargs:
                artifact = tool_msg.additional_kwargs['artifact']
            
            if artifact is None:
                logger.debug(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Инструмент {tool_name} не вернул artifact"
                )
                continue
            
            # Обрабатываем artifact (может быть списком или одиночным значением)
            try:
                if isinstance(artifact, list):
                    # Если artifact - список, извлекаем все ID
                    for item in artifact:
                        if isinstance(item, (int, str)):
                            product_id = int(item)
                            if product_id > 0:
                                all_product_ids.append(product_id)
                        elif isinstance(item, dict) and 'id' in item:
                            # Если элемент - словарь с ключом 'id'
                            product_id = int(item['id'])
                            if product_id > 0:
                                all_product_ids.append(product_id)
                elif isinstance(artifact, (int, str)):
                    # Если artifact - одиночное значение
                    product_id = int(artifact)
                    if product_id > 0:
                        all_product_ids.append(product_id)
                elif isinstance(artifact, dict):
                    # Если artifact - словарь, пытаемся извлечь ID
                    if 'id' in artifact:
                        product_id = int(artifact['id'])
                        if product_id > 0:
                            all_product_ids.append(product_id)
                    elif 'product_ids' in artifact:
                        # Если artifact содержит список product_ids
                        ids_list = artifact['product_ids']
                        if isinstance(ids_list, list):
                            for item in ids_list:
                                product_id = int(item)
                                if product_id > 0:
                                    all_product_ids.append(product_id)
                
                logger.debug(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Извлечено {len(all_product_ids)} product_ids из инструмента {tool_name}"
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Ошибка извлечения product_ids из artifact инструмента {tool_name}: {e}. "
                    f"Artifact type: {type(artifact)}, value: {artifact}"
                )
                continue
        
        # Сохраняем найденные product_ids в контекст
        if all_product_ids:
            try:
                # Удаляем дубликаты, сохраняя порядок
                unique_ids = list(dict.fromkeys(all_product_ids))
                logger.info(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Найдено {len(unique_ids)} уникальных product_ids для {client_phone}"
                )
                
                # Валидируем ID товаров (проверяем существование в БД)
                validated_ids = await self._validate_product_ids(unique_ids)
                
                if validated_ids:
                    await save_product_ids_to_context(client_phone, validated_ids)
                    logger.info(
                        f"[ProductAgent._extract_and_save_product_ids] "
                        f"Сохранено {len(validated_ids)} валидных product_ids в контекст для {client_phone}"
                    )
                    
                    invalid_ids = set(unique_ids) - set(validated_ids)
                    if invalid_ids:
                        logger.warning(
                            f"[ProductAgent._extract_and_save_product_ids] "
                            f"Пропущены несуществующие product_ids: {sorted(invalid_ids)}"
                        )
                else:
                    logger.warning(
                        f"[ProductAgent._extract_and_save_product_ids] "
                        f"Не найдено валидных product_ids для {client_phone}"
                    )
            except Exception as e:
                logger.error(
                    f"[ProductAgent._extract_and_save_product_ids] "
                    f"Ошибка сохранения product_ids для {client_phone}: {e}",
                    exc_info=True
                )

    async def _validate_product_ids(self, product_ids: List[int]) -> List[int]:
        """Валидирует список ID товаров, проверяя их существование в БД.
        
        Args:
            product_ids: Список ID товаров для валидации
            
        Returns:
            Список валидных ID товаров, существующих в БД
        """
        if not product_ids:
            return []
        
        # Лимит на количество ID для валидации (защита от слишком больших списков)
        MAX_VALIDATION_IDS = 1000
        if len(product_ids) > MAX_VALIDATION_IDS:
            logger.warning(
                f"[ProductAgent._validate_product_ids] "
                f"Список product_ids слишком большой ({len(product_ids)} > {MAX_VALIDATION_IDS}), "
                f"валидируем только первые {MAX_VALIDATION_IDS}"
            )
            product_ids = product_ids[:MAX_VALIDATION_IDS]
        
        try:
            from src.utils.supabase_client import get_supabase_client
            
            supabase = await get_supabase_client()
            validated_ids = []
            batch_size = 100
            
            for i in range(0, len(product_ids), batch_size):
                batch = product_ids[i:i + batch_size]
                try:
                    result = (
                        await supabase.table("products")
                        .select("id")
                        .in_("id", batch)
                        .execute()
                    )
                    
                    if result.data:
                        validated_ids.extend(row["id"] for row in result.data if row.get("id"))
                except Exception as e:
                    logger.warning(
                        f"[ProductAgent._validate_product_ids] "
                        f"Ошибка валидации батча {i}-{i+batch_size}: {e}"
                    )
            
            return validated_ids
        except Exception as e:
            logger.error(
                f"[ProductAgent._validate_product_ids] Ошибка валидации: {e}",
                exc_info=True
            )
            return []

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
                    response_text = " ".join(text_parts) or str(content)
                else:
                    response_text = str(content) or ""
                break

        if not response_text:
            response_text = result.get("output", "")
        
        response_text = self._postprocess_response(response_text)
        
        if not response_text or len(response_text.strip()) < 3:
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
            if not is_memory_initialized(self.memory):
                logger.warning(f"[ProductAgent] Память не инициализирована для {client_phone}, пропускаем сохранение")
                return

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
            try:
                temperature = await get_rule_as_float("DEFAULT_TEMPERATURE")
                temperature = max(0.0, min(2.0, temperature))
                if hasattr(self.llm, 'temperature'):
                    self.llm.temperature = temperature
            except Exception:
                pass
            
            try:
                seed = await get_rule_as_int("LLM_SEED")
                if hasattr(self.llm, 'seed'):
                    self.llm.seed = seed
            except Exception:
                pass

            base_prompt, system_vars, client_info, chat_history = await self._load_prompt_and_context(
                topic, client_phone
            )

            instruction_rules = await get_all_instruction_rules()
            
            if instruction_rules:
                instructions = []
                for rule_name, rule_value in instruction_rules.items():
                    if rule_value:
                        display_name = rule_name.replace("_", " ").title()
                        instructions.append(f"--- {display_name} ---\n{rule_value}")
                
                if instructions:
                    instructions_text = "\n\n".join(instructions)
                    base_prompt = f"{base_prompt}\n\n{instructions_text}"

            final_prompt = build_prompt_with_context(
                base_prompt=base_prompt,
                client_info=client_info,
                system_vars=system_vars if system_vars else None,
            )
            self.SYSTEM_PROMPT = final_prompt

            input_with_context = self._prepare_messages(user_input, chat_history)

            from .tools.context_tools import get_agent_context_async
            await get_agent_context_async(client_phone)
            
            client_phone_context.set(client_phone)
            
            context_tools = [set_photo_requirement, get_conversation_context]
            sql_tools = create_sql_tools()
            media_tools = [show_product_photos, send_pricelist]
            
            agent_tools = self.tools + sql_tools + media_tools + context_tools

            callbacks_list = [
                langfuse_handler,
                StdOutCallbackHandler(),
            ]

            try:
                recursion_limit = await get_rule_as_int("AGENT_RECURSION_LIMIT")
                recursion_limit = max(1, min(10000, recursion_limit))
            except Exception:
                recursion_limit = 1005
            
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

            messages = []
            if chat_history:
                messages.extend(chat_history)
            messages.append(HumanMessage(content=input_with_context))

            try:
                result = await self._execute_agent(messages, agent_tools, config)
            except asyncio.TimeoutError:
                try:
                    max_execution_time = await get_rule_as_int("MAX_AGENT_EXECUTION_TIME")
                except Exception:
                    max_execution_time = 3600
                error_msg = f"Агент превысил максимальное время выполнения ({max_execution_time} секунд)"
                logger.error(f"[ProductAgent.run] Timeout агента: {error_msg}")
                raise Exception(error_msg)
            except Exception as e:
                error_msg = f"Ошибка при выполнении агента: {str(e)}"
                logger.error(f"[ProductAgent.run] Ошибка агента: {error_msg}", exc_info=True)
                raise Exception(error_msg) from e
            finally:
                client_phone_context.set('')

            response_text = self._extract_response(result)

            # Извлекаем product_ids из artifacts инструментов поиска товаров
            # Примечание: product_ids уже должны быть сохранены через middleware во время выполнения,
            # но оставляем этот вызов как fallback на случай, если middleware не сработал
            await self._extract_and_save_product_ids(result, client_phone)

            await self._save_to_memory(user_input, response_text, chat_history, client_phone)

            langfuse_handler.save_conversation_to_langfuse()

            return response_text

        except Exception as e:
            from src.config.messages_constants import ERROR_MESSAGE_AGENT_FAILED
            error_msg = ERROR_MESSAGE_AGENT_FAILED
            logger.error(f"[ProductAgent.run] Ошибка ProductAgent: {str(e)}", exc_info=True)

            try:
                langfuse_handler.save_conversation_to_langfuse()
            except Exception as langfuse_error:
                logger.warning(f"Не удалось сохранить ошибку в LangFuse: {langfuse_error}")

            return error_msg