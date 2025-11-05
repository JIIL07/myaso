"""ProductAgent - агент для работы с продуктами и продажами.

Использует LangChain AgentExecutor для обработки запросов пользователей
с использованием tools, памяти и профиля клиента.
"""

from __future__ import annotations

from typing import Any, List, Optional
import hashlib
import logging
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManager
from langsmith import Client
from langchain_core.tracers import LangChainTracer

from src.config.settings import settings
from src.config.langchain_settings import LangChainSettings
from .base_agent import BaseAgent
from agents.tools import (
    enhance_user_product_query,
    show_product_photos,
    get_client_profile,
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
- Поиска товаров по запросу клиента (enhance_user_product_query)
- Отправки фотографий товаров (show_product_photos)
- Получения информации о профиле клиента (get_client_profile)

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
        if langchain_settings.langsmith_tracing_enabled and langchain_settings.langsmith_api_key:
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
                temperature=0.8,
                callbacks=callbacks,
            )

        if tools is None:
            tools = [enhance_user_product_query, show_product_photos, get_client_profile]

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.agent_type = agent_type
        self._agent_executor: Optional[AgentExecutor] = None
        self._callbacks = callbacks
        self.SYSTEM_PROMPT = self.DEFAULT_SYSTEM_PROMPT
        self._last_prompt_hash: Optional[str] = None

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
            max_iterations=5,
            max_execution_time=30,
            callbacks=self._callbacks,
        )

        return agent_executor

    async def run(self, user_input: str, client_phone: str, topic: Optional[str] = None) -> str:
        """Запускает агента для обработки запроса пользователя.

        Args:
            user_input: Текст запроса пользователя
            client_phone: Номер телефона клиента
            topic: Тема диалога для загрузки промпта из БД (опционально)

        Returns:
            Строка с ответом агента

        Raises:
            Exception: При ошибке выполнения агента
        """
        logger.info(f"[ProductAgent.run] Начало выполнения для {client_phone}, topic: {topic}")
        try:
            db_prompt = None
            if topic:
                try:
                    db_prompt = await get_prompt(topic)
                    if db_prompt:
                        logger.info(f"Загружен промпт из БД для topic '{topic}'")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить промпт для topic '{topic}': {e}")
            
            system_vars = {}
            try:
                system_vars = await get_all_system_values()
                if system_vars:
                    logger.info(f"Загружено системных переменных: {len(system_vars)}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить системные переменные: {e}")
            
            profile_context = ""
            try:
                profile_result = await get_client_profile.ainvoke({"phone": client_phone})
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
                system_vars_text = "\n".join([f"{k}: {v}" for k, v in system_vars.items()])
                final_prompt = f"{self.DEFAULT_SYSTEM_PROMPT}\n\nСистемные переменные:\n{system_vars_text}"
            else:
                final_prompt = self.DEFAULT_SYSTEM_PROMPT
            
            prompt_hash = hashlib.md5(final_prompt.encode()).hexdigest()
            if self._last_prompt_hash != prompt_hash:
                self.SYSTEM_PROMPT = final_prompt
                self._last_prompt_hash = prompt_hash
                self._agent_executor = None
                logger.info("Промпт изменился, пересоздаем AgentExecutor")
            
            if self._agent_executor is None:
                self._agent_executor = self._create_agent_executor()

            chat_history: List[BaseMessage] = []
            if self.memory is not None:
                try:
                    memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
                    chat_history = memory_vars.get("history", [])
                except Exception as e:
                    logger.warning(f"Не удалось загрузить память: {e}")
                    chat_history = []
            
            input_with_context = user_input
            full_prompt_parts = ["=== ПОЛНЫЙ ПРОМПТ К LLM ===\n"]
            full_prompt_parts.append(f"System:\n{self.SYSTEM_PROMPT}\n")
            
            if chat_history:
                full_prompt_parts.append(f"Chat History ({len(chat_history)} сообщений):")
                for i, msg in enumerate(chat_history, 1):
                    if isinstance(msg, HumanMessage):
                        full_prompt_parts.append(f"  [{i}] Human: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        full_prompt_parts.append(f"  [{i}] AI: {msg.content}")
                    elif isinstance(msg, SystemMessage):
                        full_prompt_parts.append(f"  [{i}] System: {msg.content}")
            else:
                full_prompt_parts.append("Chat History: (пусто)")
            
            full_prompt_parts.append(f"\nUser Input:\n{input_with_context}\n")
            full_prompt_parts.append("=" * 50)
            
            full_prompt_text = "\n".join(full_prompt_parts)
            logger.info(full_prompt_text)

            try:
                result = await self._agent_executor.ainvoke(
                    {
                        "input": input_with_context,
                        "chat_history": chat_history,
                    }
                )
            except Exception as e:
                error_msg = f"Ошибка при выполнении агента: {str(e)}"
                logger.error(f"AgentExecutor error: {error_msg}", exc_info=True)
                raise Exception(error_msg) from e

            response_text = result.get("output", "")
            if not response_text:
                response_text = "Упс, что-то пошло не так 😅. Попробуйте переформулировать запрос, и я обязательно помогу!"

            if self.memory is not None:
                try:
                    await self.memory.add_messages([HumanMessage(content=user_input)])
                    await self.memory.add_messages([AIMessage(content=response_text)])
                except Exception as e:
                    logger.warning(f"Не удалось сохранить в память: {e}")

            logger.info(f"[ProductAgent.run] Завершение выполнения для {client_phone}, длина ответа: {len(response_text)}")
            return response_text

        except Exception as e:
            error_msg = f"Ой, что-то пошло не так 😔. Попробуйте написать еще раз, пожалуйста!"
            logger.error(f"ProductAgent error: {str(e)}", exc_info=True)
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
