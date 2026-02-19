import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from src.entities import (
    InitConversationRequest,
    ResetConversationRequest,
    UserMessageRequest,
)
from src.services.ai.prompt import compose_prompts
from src.services.agent_queue import QueueManager, RateLimiter, StatusService, AgentQueueWorker
from src.services.queue.queue import send_delayed_message
from src.utils.prompts import get_langfuse_label
from src.constants import (
    PROMPT_NAME_COORDINATOR,
    PROMPT_NAME_FUNCTION,
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
    _instance: Optional["ConversationService"] = None

    def __init__(self) -> None:
        from src.agent.product_agent.factory import AgentFactory

        self.factory = AgentFactory.instance()
        self._messaging_service = None
        self._queue_manager = QueueManager()
        self._rate_limiter = RateLimiter(max_concurrent=1)
        self._status_service = StatusService(self._queue_manager, self._rate_limiter)
        self._worker = AgentQueueWorker(
            self._queue_manager,
            self._rate_limiter,
            self._status_service,
        )
        self._worker.register_handler("process", self._handle_process_task)
        self._worker.register_handler("init", self._handle_init_task)
        self._worker_task: Optional[asyncio.Task] = None

    @classmethod
    def instance(cls) -> "ConversationService":
        """Return the shared ConversationService singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_messaging_service(self):
        if self._messaging_service is None:
            from src.services.telegram.telegram import TelegramMessagingService
            self._messaging_service = TelegramMessagingService()
        return self._messaging_service

    async def _ensure_worker_running(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker.start())
            logger.info("[Conversation] Worker started")

    async def _handle_process_task(
        self, client_phone: str, message: str, message_received_time: datetime
    ) -> dict[str, Any]:
        request = UserMessageRequest(client_phone=client_phone, message=message)
        return await self._process_conversation_internal(request, message_received_time)

    async def _handle_init_task(
        self, client_phone: str, message: str, message_received_time: datetime
    ) -> dict[str, Any]:
        request = InitConversationRequest(client_phone=client_phone)
        return await self._init_conversation_internal(request, message_received_time)

    async def _process_conversation_internal(
        self, request: UserMessageRequest, message_received_time: datetime
    ) -> dict[str, Any]:
        context = "processConversation"
        try:
            _, agent = await self._create_agent_with_memory(
                client_phone=request.client_phone, clear_memory=False,
            )
            system_prompt = await self._build_process_conversation_prompt(request.client_phone)
            response_text = await agent.run(
                user_input=request.message,
                client_phone=request.client_phone,
                endpoint_name=context,
                system_prompt=system_prompt,
            )
            await self._schedule_delayed_reply(context, request.client_phone, response_text, message_received_time)
            return {"success": True, "response_text": response_text}
        except Exception as e:
            logger.error("[%s] Error processing for %s: %s", context, request.client_phone, e, exc_info=True)
            return {"success": False, "error": str(e)}

    async def _init_conversation_internal(
        self, request: InitConversationRequest, message_received_time: datetime
    ) -> dict[str, Any]:
        context = "initConversation"
        try:
            _, agent = await self._create_agent_with_memory(
                client_phone=request.client_phone, clear_memory=True,
            )
            system_prompt = await self._build_init_conversation_prompt(request.client_phone)
            response_text = await agent.run(
                user_input=system_prompt,
                client_phone=request.client_phone,
                endpoint_name=context,
                system_prompt=system_prompt,
            )
            await self._schedule_delayed_reply(context, request.client_phone, response_text, message_received_time)
            return {"success": True, "response_text": response_text}
        except Exception as e:
            logger.error("[%s] Critical error for %s: %s", context, request.client_phone, e, exc_info=True)
            return {"success": False, "error": str(e)}

    async def _initialize_memory(self, client_phone: str) -> SupabaseConversationMemory:
        return await SupabaseConversationMemory.create(client_phone)

    async def _create_agent_with_memory(self, client_phone: str, clear_memory: bool = False):
        memory = await self._initialize_memory(client_phone)
        if clear_memory:
            await memory.clear()
        agent = self.factory.create_product_agent(config={"memory": memory}, use_cache=False)
        return memory, agent

    async def _get_client_style(self, client_phone: str) -> Optional[str]:
        try:
            return await get_client_style(client_phone)
        except Exception as e:
            logger.warning("[Conversation] Error getting style for %s: %s", client_phone, e)
            return None

    async def _get_style_prompt_name(self, style: Optional[str]) -> Optional[str]:
        if not style:
            return None
        style_map = {
            "эдуард": PROMPT_NAME_STYLE_EDUARD,
            "полина": PROMPT_NAME_STYLE_POLINA,
            "маша": PROMPT_NAME_STYLE_MASHA,
        }
        result = style_map.get(style.lower().strip())
        if result is None:
            logger.warning("[Conversation] Unknown conversation style: %s", style)
        return result

    async def _build_init_conversation_prompt(self, client_phone: str) -> str:
        langfuse_label = get_langfuse_label()
        prompt_names = [
            PROMPT_NAME_SYSTEM_PROMPT, PROMPT_NAME_PROFILE,
            PROMPT_NAME_PRODUCTS, PROMPT_NAME_OFFER,
        ]
        style = await self._get_client_style(client_phone)
        style_prompt = await self._get_style_prompt_name(style)
        if style_prompt:
            prompt_names.append(style_prompt)
        prompt_names.append(PROMPT_NAME_REFLECTOR)
        return await compose_prompts(
            prompt_names=prompt_names, separator="\n\n",
            langfuse_label=langfuse_label,
            variables={"client_phone": client_phone},
            context="initConversation",
        )

    async def _build_process_conversation_prompt(self, client_phone: str) -> str:
        langfuse_label = get_langfuse_label()
        return await compose_prompts(
            prompt_names=[
                PROMPT_NAME_SYSTEM_PROMPT, PROMPT_NAME_PROFILE,
                PROMPT_NAME_COORDINATOR, PROMPT_NAME_FUNCTION,
                PROMPT_NAME_INFO, PROMPT_NAME_UNCLEAR,
            ],
            separator="\n\n",
            langfuse_label=langfuse_label,
            variables={"client_phone": client_phone},
            context="processConversation",
        )

    async def _schedule_delayed_reply(
        self, context: str, client_phone: str, response_text: str, message_received_time: datetime,
    ) -> None:
        elapsed_time = (datetime.now() - message_received_time).total_seconds()
        delay_seconds = max(0, int(900 - elapsed_time))
        msg_id = await send_delayed_message(client_phone=client_phone, message=response_text, delay=delay_seconds)
        if msg_id is None:
            logger.warning("[%s] Failed to add response to PGMQ for %s", context, client_phone)

    async def process_conversation(self, request: UserMessageRequest) -> dict[str, Any]:
        context = "processConversation"
        message_received_time = datetime.now()
        await self._ensure_worker_running()

        if self._rate_limiter.is_available():
            logger.info("[%s] Agent available, processing for %s", context, request.client_phone)
            return await self._process_conversation_internal(request, message_received_time)

        logger.info("[%s] Agent busy, queuing for %s", context, request.client_phone)
        await self._queue_manager.add_task(
            client_phone=request.client_phone,
            message=request.message,
            task_type="process",
            message_received_time=message_received_time,
        )
        return {"success": True, "queued": True}

    async def init_conversation(self, request: InitConversationRequest) -> dict[str, Any]:
        context = "initConversation"
        message_received_time = datetime.now()
        await self._ensure_worker_running()

        if self._rate_limiter.is_available():
            logger.info("[%s] Agent available, processing for %s", context, request.client_phone)
            return await self._init_conversation_internal(request, message_received_time)

        logger.info("[%s] Agent busy, queuing for %s", context, request.client_phone)
        await self._queue_manager.add_task(
            client_phone=request.client_phone,
            message="",
            task_type="init",
            message_received_time=message_received_time,
        )
        return {"success": True, "queued": True}

    async def reset_conversation(self, request: ResetConversationRequest) -> dict[str, Any]:
        context = "resetConversation"
        try:
            memory = await SupabaseConversationMemory.create(request.client_phone)
            await memory.clear()
            logger.info("[%s] History reset for %s", context, request.client_phone)
            return {"success": True}
        except Exception as e:
            logger.error("[%s] Error for %s: %s", context, request.client_phone, e, exc_info=True)
            return {"success": False}

    async def process_conversation_async(self, client_phone: str, message: str) -> dict[str, Any]:
        request = UserMessageRequest(client_phone=client_phone, message=message)
        return await self.process_conversation(request)

    async def init_conversation_async(self, client_phone: str) -> dict[str, Any]:
        request = InitConversationRequest(client_phone=client_phone)
        return await self.init_conversation(request)

    async def reset_conversation_async(self, client_phone: str) -> dict[str, Any]:
        request = ResetConversationRequest(client_phone=client_phone)
        return await self.reset_conversation(request)
