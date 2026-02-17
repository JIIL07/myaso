"""Сервис для управления разговорами с клиентами."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from src.entities import (
    InitConversationRequest,
    ResetConversationRequest,
    UserMessageRequest,
)

from src.agent.product_agent.factory import AgentFactory
from src.services.ai.exceptions import ConversationError
from src.services.ai.prompt import compose_prompts, get_prompt
from src.services.agent_queue import QueueManager, RateLimiter, StatusService, AgentQueueWorker
from src.services.queue.queue import send_delayed_message
from src.utils.prompts import get_langfuse_label
from src.services.langfuse.prompt_names import (
    PROMPT_NAME_COORDINATOR,
    PROMPT_NAME_ERROR_HANDLER,
    PROMPT_NAME_FUNCTION,
    PROMPT_NAME_HUMAN_IN_THE_LOOP,
    PROMPT_NAME_INFO,
    PROMPT_NAME_OFFER,
    PROMPT_NAME_PRODUCTS,
    PROMPT_NAME_PROFILE,
    PROMPT_NAME_REFLECTOR,
    PROMPT_NAME_STYLE_EDUARD,
    PROMPT_NAME_STYLE_MASHA,
    PROMPT_NAME_STYLE_POLINA,
    PROMPT_NAME_SYSTEM_PROMPT,
    PROMPT_NAME_UNCLEAR,
)
from src.services.memory import SupabaseConversationMemory
from src.queries.clients_queries import get_client_style

logger = logging.getLogger(__name__)


class ConversationService:
    """Управляет разговорами с AI-агентом и очередью обработки."""

    def __init__(self):
        self.factory = AgentFactory.instance()
        self._messaging_service = None
        
        # Инициализация сервисов очереди (singleton)
        self._queue_manager = QueueManager()
        self._rate_limiter = RateLimiter(max_concurrent=1)
        self._status_service = StatusService(self._queue_manager, self._rate_limiter)
        self._worker = AgentQueueWorker(
            self._queue_manager,
            self._rate_limiter,
            self._status_service,
        )
        
        # Регистрация обработчиков для worker
        self._worker.register_handler("process", self._handle_process_task)
        self._worker.register_handler("init", self._handle_init_task)
        
        # Запуск worker в фоне (если еще не запущен)
        self._worker_task: Optional[asyncio.Task] = None

    def _get_messaging_service(self):
        """Ленивая инициализация TelegramMessagingService для избежания циклических импортов."""
        if self._messaging_service is None:
            from src.services.telegram.telegram import TelegramMessagingService
            self._messaging_service = TelegramMessagingService()
        return self._messaging_service

    async def _ensure_worker_running(self) -> None:
        """Обеспечивает запуск worker для обработки очереди."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker.start())
            logger.info("[ConversationService] Worker запущен")

    async def _handle_process_task(
        self, client_phone: str, message: str, message_received_time: datetime
    ) -> Dict[str, Any]:
        """Обрабатывает задачу process из очереди.

        Args:
            client_phone: Номер телефона клиента
            message: Текст сообщения
            message_received_time: Время получения сообщения

        Returns:
            Результат обработки
        """
        request = UserMessageRequest(client_phone=client_phone, message=message)
        result = await self._process_conversation_internal(request, message_received_time)
        return result

    async def _handle_init_task(
        self, client_phone: str, message: str, message_received_time: datetime
    ) -> Dict[str, Any]:
        """Обрабатывает задачу init из очереди.

        Args:
            client_phone: Номер телефона клиента
            message: Текст сообщения (не используется для init)
            message_received_time: Время получения запроса

        Returns:
            Результат обработки
        """
        request = InitConversationRequest(client_phone=client_phone)
        result = await self._init_conversation_internal(request, message_received_time)
        return result

    async def _process_conversation_internal(
        self, request: UserMessageRequest, message_received_time: datetime
    ) -> Dict[str, Any]:
        """Внутренний метод обработки сообщения (используется worker'ом).

        Args:
            request: Запрос с сообщением пользователя
            message_received_time: Время получения сообщения

        Returns:
            Словарь с результатом обработки
        """
        context = "processConversation"

        try:
            _, agent = await self._create_agent_with_memory(
                client_phone=request.client_phone,
                clear_memory=False,
            )

            system_prompt = await self._build_process_conversation_prompt(
                client_phone=request.client_phone,
            )

            user_input = await self._prepare_user_input(
                request.message, add_tool_instruction=True
            )

            response_text = await agent.run(
                user_input=user_input,
                client_phone=request.client_phone,
                endpoint_name=context,
                system_prompt=system_prompt,
            )

            await self._schedule_delayed_reply(
                context=context,
                client_phone=request.client_phone,
                response_text=response_text,
                message_received_time=message_received_time,
            )

            return {"success": True, "response_text": response_text}

        except Exception as e:
            logger.error(
                f"[{context}] Ошибка обработки для {request.client_phone}: {e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e)}

    async def _init_conversation_internal(
        self, request: InitConversationRequest, message_received_time: datetime
    ) -> Dict[str, Any]:
        """Внутренний метод инициализации беседы (используется worker'ом).

        Args:
            request: Запрос с номером телефона клиента
            message_received_time: Время получения запроса

        Returns:
            Словарь с результатом инициализации
        """
        context = "initConversation"

        try:
            _, agent = await self._create_agent_with_memory(
                client_phone=request.client_phone,
                clear_memory=True,
            )

            system_prompt = await self._build_init_conversation_prompt(
                client_phone=request.client_phone,
            )

            user_input = system_prompt

            response_text = await agent.run(
                user_input=user_input,
                client_phone=request.client_phone,
                endpoint_name=context,
                system_prompt=system_prompt,
            )

            await self._schedule_delayed_reply(
                context=context,
                client_phone=request.client_phone,
                response_text=response_text,
                message_received_time=message_received_time,
            )

            return {"success": True, "response_text": response_text}

        except Exception as e:
            logger.error(
                f"[{context}] Критическая ошибка для {request.client_phone}: {e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e)}

    async def _initialize_memory(self, client_phone: str) -> SupabaseConversationMemory:
        """Инициализирует память для разговора.

        Args:
            client_phone: Номер телефона клиента

        Returns:
            Инициализированный объект памяти
        """
        memory = await SupabaseConversationMemory(client_phone)
        return memory

    async def _create_agent_with_memory(
        self,
        client_phone: str,
        clear_memory: bool = False,
    ):
        """Создает агента с инициализированной памятью.

        Args:
            client_phone: Номер телефона клиента
            clear_memory: Нужно ли очистить память перед использованием

        Returns:
            Кортеж (memory, agent)
        """
        memory = await self._initialize_memory(client_phone)
        if clear_memory:
            await memory.clear()

        agent = self.factory.create_product_agent(
            config={"memory": memory},
            use_cache=False,
        )

        return memory, agent

    async def _get_client_style(self, client_phone: str) -> Optional[str]:
        """Получает стиль общения клиента из БД.

        Args:
            client_phone: Номер телефона клиента

        Returns:
            Стиль общения ('Эдуард', 'Полина', 'Маша') или None
        """
        try:
            style = await get_client_style(client_phone)
            return style
        except Exception as e:
            logger.warning(
                f"[_get_client_style] Ошибка при получении стиля из БД для {client_phone}: {e}"
            )
            return None

    async def _get_style_prompt_name(self, style: Optional[str]) -> Optional[str]:
        """Получает название промпта стиля общения.

        Args:
            style: Название стиля ('Эдуард', 'Полина', 'Маша')

        Returns:
            Название промпта стиля или None
        """
        if not style:
            return None
        
        style_lower = style.lower().strip()
        if style_lower == "эдуард":
            return PROMPT_NAME_STYLE_EDUARD
        elif style_lower == "полина":
            return PROMPT_NAME_STYLE_POLINA
        elif style_lower == "маша":
            return PROMPT_NAME_STYLE_MASHA
        
        logger.warning(f"Неизвестный стиль общения: {style}, используем дефолтный")
        return None

    async def _build_init_conversation_prompt(
        self,
        client_phone: str,
    ) -> str:
        """Собирает промпт для инициализации беседы.

        Порядок загрузки промптов:
        1. Системный промт (базовый)
        2. Профиль (анализ профиля клиента) + автоматически полученная информация о клиенте
        3. Товары (действия из промпта)
        4. Предложение (действия из промпта)
        5. Стиль общения (если указан в БД)
        6. Рефлектор (действия из промпта)

        Args:
            client_phone: Номер телефона клиента (идентификатор клиента)

        Returns:
            Собранный промпт
        """
        langfuse_label = get_langfuse_label()

        prompt_names = [
            PROMPT_NAME_SYSTEM_PROMPT,
            PROMPT_NAME_PROFILE,
            PROMPT_NAME_PRODUCTS,
            PROMPT_NAME_OFFER,
        ]

        # Получаем стиль из БД
        style = await self._get_client_style(client_phone)
        style_prompt = await self._get_style_prompt_name(style)
        if style_prompt:
            prompt_names.append(style_prompt)

        prompt_names.append(PROMPT_NAME_REFLECTOR)

        composed_prompt = await compose_prompts(
            prompt_names=prompt_names,
            separator="\n\n",
            langfuse_label=langfuse_label,
            variables={"client_phone": client_phone},
            context="initConversation",
        )

        return composed_prompt

    async def _build_process_conversation_prompt(
        self,
        client_phone: str,
    ) -> str:
        """Собирает промпт для обработки сообщения.

        Порядок загрузки промптов:
        1. Системный промт (базовый)
        2. Профиль (анализ профиля клиента)
        3. Координатор (действия из промпта)
        4. function, info, unclear (всегда добавляются в конец)

        Args:
            client_phone: Номер телефона клиента

        Returns:
            Собранный промпт
        """
        langfuse_label = get_langfuse_label()

        prompt_names = [
            PROMPT_NAME_SYSTEM_PROMPT,
            PROMPT_NAME_PROFILE,
            PROMPT_NAME_COORDINATOR,
            PROMPT_NAME_FUNCTION,
            PROMPT_NAME_INFO,
            PROMPT_NAME_UNCLEAR,
        ]

        composed_prompt = await compose_prompts(
            prompt_names=prompt_names,
            separator="\n\n",
            langfuse_label=langfuse_label,
            variables={"client_phone": client_phone},
            context="processConversation",
        )

        return composed_prompt

    async def _schedule_delayed_reply(
        self,
        context: str,
        client_phone: str,
        response_text: str,
        message_received_time: datetime,
    ) -> None:
        """Отправляет ответ в pqmq с задержкой 15 минут от момента получения сообщения."""
        elapsed_time = (datetime.now() - message_received_time).total_seconds()
        delay_seconds = max(0, int(900 - elapsed_time))  # 15 минут = 900 секунд

        msg_id = await send_delayed_message(
            client_phone=client_phone,
            message=response_text,
            delay=delay_seconds,
        )

        if msg_id is None:
            logger.warning(
                f"[{context}] Не удалось добавить ответ в pqmq для {client_phone}"
            )

    async def _prepare_user_input(
        self, user_input: str, add_tool_instruction: bool = False
    ) -> str:
        """Возвращает пользовательский ввод без добавления inline-инструкций."""
        return user_input

    async def process_conversation(self, request: UserMessageRequest) -> Dict[str, Any]:
        """Обрабатывает сообщение пользователя.

        Проверяет доступность агента. Если свободен - обрабатывает сразу,
        если занят - добавляет в очередь.

        Args:
            request: Запрос с сообщением пользователя

        Returns:
            Словарь с результатом обработки
        """
        context = "processConversation"
        message_received_time = datetime.now()

        # Обеспечиваем запуск worker
        await self._ensure_worker_running()

        # Проверяем доступность агента
        if self._rate_limiter.is_available():
            # Агент свободен - обрабатываем сразу
            logger.info(
                f"[{context}] Агент свободен, обрабатываем сразу для {request.client_phone}"
            )
            return await self._process_conversation_internal(request, message_received_time)
        else:
            # Агент занят - добавляем в очередь
            logger.info(
                f"[{context}] Агент занят, добавляем в очередь для {request.client_phone}"
            )
            await self._queue_manager.add_task(
                client_phone=request.client_phone,
                message=request.message,
                task_type="process",
                message_received_time=message_received_time,
            )
            return {"success": True, "queued": True}

    async def init_conversation(
        self, request: InitConversationRequest
    ) -> Dict[str, Any]:
        """Инициализирует новую беседу с клиентом.

        Проверяет доступность агента. Если свободен - обрабатывает сразу,
        если занят - добавляет в очередь.

        Args:
            request: Запрос с номером телефона клиента

        Returns:
            Словарь с результатом инициализации
        """
        context = "initConversation"
        message_received_time = datetime.now()

        # Обеспечиваем запуск worker
        await self._ensure_worker_running()

        # Проверяем доступность агента
        if self._rate_limiter.is_available():
            # Агент свободен - обрабатываем сразу
            logger.info(
                f"[{context}] Агент свободен, обрабатываем сразу для {request.client_phone}"
            )
            return await self._init_conversation_internal(request, message_received_time)
        else:
            # Агент занят - добавляем в очередь
            logger.info(
                f"[{context}] Агент занят, добавляем в очередь для {request.client_phone}"
            )
            await self._queue_manager.add_task(
                client_phone=request.client_phone,
                message="",  # Для init сообщение не используется
                task_type="init",
                message_received_time=message_received_time,
            )
            return {"success": True, "queued": True}

    def get_queue_status(self) -> Dict[str, Any]:
        """Возвращает статус очереди и агента.

        Returns:
            Словарь со статусом очереди и агента
        """
        return self._status_service.get_status()

    async def process_conversation_test(self, request: UserMessageRequest) -> Dict[str, Any]:
        """Тестовая версия обработки сообщения пользователя без отправки уведомлений.

        Args:
            request: Запрос с сообщением пользователя

        Returns:
            Словарь с результатом обработки: success, response_text, error
        """
        context = "processConversationTest"

        try:
            _, agent = await self._create_agent_with_memory(
                client_phone=request.client_phone,
                clear_memory=False,
            )

            system_prompt = await self._build_process_conversation_prompt(
                client_phone=request.client_phone,
            )

            user_input = await self._prepare_user_input(
                request.message, add_tool_instruction=True
            )

            response_text = await agent.run(
                user_input=user_input,
                client_phone=request.client_phone,
                endpoint_name=context,
                system_prompt=system_prompt,
            )

            return {
                "success": True,
                "response_text": response_text,
            }

        except ConversationError as e:
            logger.error(f"[{context}] Ошибка конверсации: {e}")
            return {
                "success": False,
                "error": str(e),
            }

        except Exception as e:
            logger.error(
                f"[{context}] Ошибка обработки для {request.client_phone}: {e}",
                exc_info=True,
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def init_conversation_test(
        self, request: InitConversationRequest
    ) -> Dict[str, Any]:
        """Тестовая версия инициализации беседы без отправки уведомлений.

        Args:
            request: Запрос с номером телефона клиента и стилем общения

        Returns:
            Словарь с результатом инициализации: success, response_text, error
        """
        context = "initConversationTest"

        try:
            _, agent = await self._create_agent_with_memory(
                client_phone=request.client_phone,
                clear_memory=True,
            )

            system_prompt = await self._build_init_conversation_prompt(
                client_phone=request.client_phone,
            )

            user_input = system_prompt

            response_text = await agent.run(
                user_input=user_input,
                client_phone=request.client_phone,
                endpoint_name=context,
                system_prompt=system_prompt,
            )

            return {
                "success": True,
                "response_text": response_text,
            }

        except ConversationError as e:
            logger.error(f"[{context}] Ошибка конверсации: {e}")
            return {
                "success": False,
                "error": str(e),
            }

        except Exception as e:
            logger.error(
                f"[{context}] Критическая ошибка для {request.client_phone}: {e}",
                exc_info=True,
            )
            return {
                "success": False,
                "error": str(e),
            }

    async def reset_conversation(
        self, request: ResetConversationRequest
    ) -> Dict[str, Any]:
        """Сбрасывает историю беседы для клиента.

        Args:
            request: Запрос с номером телефона клиента

        Returns:
            Словарь с результатом сброса
        """
        context = "resetConversation"

        try:
            memory = await SupabaseConversationMemory(request.client_phone)
            await memory.clear()
            logger.info(f"[{context}] История успешно сброшена для {request.client_phone}")
            return {"success": True}

        except Exception as e:
            logger.error(
                f"[{context}] Ошибка для {request.client_phone}: {e}", exc_info=True
            )
            return {"success": False}

    async def process_conversation_async(
        self,
        client_phone: str,
        message: str,
    ) -> Dict[str, Any]:
        """Асинхронная обработка сообщения пользователя (для Background Tasks).

        Создает новый request объект внутри задачи для изоляции ресурсов.

        Args:
            client_phone: Номер телефона клиента
            message: Текст сообщения

        Returns:
            Словарь с результатом обработки
        """
        request = UserMessageRequest(
            client_phone=client_phone,
            message=message,
        )
        return await self.process_conversation(request)

    async def init_conversation_async(
        self,
        client_phone: str,
    ) -> Dict[str, Any]:
        """Асинхронная инициализация беседы (для Background Tasks).

        Создает новый request объект внутри задачи для изоляции ресурсов.

        Args:
            client_phone: Номер телефона клиента

        Returns:
            Словарь с результатом инициализации
        """
        request = InitConversationRequest(
            client_phone=client_phone,
        )
        return await self.init_conversation(request)

    async def reset_conversation_async(
        self,
        client_phone: str,
    ) -> Dict[str, Any]:
        """Асинхронный сброс истории беседы (для Background Tasks).

        Создает новый request объект внутри задачи для изоляции ресурсов.

        Args:
            client_phone: Номер телефона клиента

        Returns:
            Словарь с результатом сброса
        """
        request = ResetConversationRequest(client_phone=client_phone)
        return await self.reset_conversation(request)
