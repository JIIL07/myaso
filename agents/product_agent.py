"""ProductAgent - агент для работы с продуктами и продажами.

Использует LangChain AgentExecutor для обработки запросов пользователей
с использованием tools, памяти и профиля клиента.
"""

from __future__ import annotations

from typing import Any, List, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config.settings import settings
from .base_agent import BaseAgent
from agents.tools import (
    enhance_user_product_query,
    show_product_photos,
    get_client_profile,
)


class ProductAgent(BaseAgent):
    """Агент для обработки запросов пользователей о товарах и продажах.

    Использует AgentExecutor с tools для поиска товаров, отправки фото
    и получения профиля клиента.
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
        # Инициализируем LLM, если не передан
        if llm is None:
            llm = ChatOpenAI(
                model=settings.openrouter.model_id,
                openai_api_key=settings.openrouter.openrouter_api_key,
                openai_api_base=settings.openrouter.base_url,
                temperature=0.7,
            )

        # Используем стандартные tools, если не переданы
        if tools is None:
            tools = [enhance_user_product_query, show_product_photos, get_client_profile]

        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.agent_type = agent_type
        self._agent_executor: Optional[AgentExecutor] = None

    def _create_agent_executor(self) -> AgentExecutor:
        """Создаёт AgentExecutor с промптом и инструментами.

        Returns:
            AgentExecutor для выполнения агента
        """
        # Системный промпт для менеджера по продажам
        system_prompt = """Ты профессиональный менеджер по продажам мясной продукции.

Твоя задача:
- Помогать клиентам найти подходящие товары из ассортимента
- Предоставлять детальную информацию о товарах (цена, вес, упаковка, поставщик)
- Отправлять фотографии товаров по запросу клиента
- Учитывать профиль клиента при рекомендациях
- Быть вежливым, дружелюбным и профессиональным

Используй доступные инструменты для:
- Поиска товаров по запросу клиента (enhance_user_product_query)
- Отправки фотографий товаров (show_product_photos)
- Получения информации о профиле клиента (get_client_profile)

Всегда старайся помочь клиенту найти именно то, что он ищет."""

        # Создаём промпт с системным сообщением и историей
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
        else:  # zero-shot-react-description
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            agent = create_react_agent(self.llm, self.tools, prompt)

        # Создаём AgentExecutor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=30,
        )

        return agent_executor

    async def run(self, user_input: str, client_phone: str) -> str:
        """Запускает агента для обработки запроса пользователя.

        Args:
            user_input: Текст запроса пользователя
            client_phone: Номер телефона клиента

        Returns:
            Строка с ответом агента

        Raises:
            Exception: При ошибке выполнения агента
        """
        try:
            # Инициализируем AgentExecutor, если ещё не создан
            if self._agent_executor is None:
                self._agent_executor = self._create_agent_executor()

            # Загружаем историю через memory
            chat_history: List[BaseMessage] = []
            if self.memory is not None:
                try:
                    memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
                    chat_history = memory_vars.get("history", [])
                except Exception as e:
                    print(f"Warning: Failed to load memory: {e}")
                    chat_history = []

            # Загружаем профиль клиента через tool
            profile_context = ""
            try:
                profile_result = await get_client_profile.ainvoke({"phone": client_phone})
                if profile_result and "не найден" not in profile_result.lower():
                    profile_context = f"\n\nПрофиль клиента:\n{profile_result}\n"
            except Exception as e:
                print(f"Warning: Failed to load client profile: {e}")

            # Формируем входной запрос с контекстом профиля
            input_with_context = user_input
            if profile_context:
                input_with_context = f"{profile_context}\n{user_input}"

            # Запускаем агента
            try:
                result = await self._agent_executor.ainvoke(
                    {
                        "input": input_with_context,
                        "chat_history": chat_history,
                    }
                )
            except Exception as e:
                error_msg = f"Ошибка при выполнении агента: {str(e)}"
                print(f"AgentExecutor error: {error_msg}")
                raise Exception(error_msg) from e

            # Извлекаем ответ из результата
            response_text = result.get("output", "")
            if not response_text:
                response_text = "Извините, произошла ошибка при обработке запроса."

            # Сохраняем сообщения в memory
            if self.memory is not None:
                try:
                    # Сохраняем сообщение пользователя
                    await self.memory.add_messages([HumanMessage(content=user_input)])
                    # Сохраняем ответ агента
                    await self.memory.add_messages([AIMessage(content=response_text)])
                except Exception as e:
                    print(f"Warning: Failed to save to memory: {e}")

            return response_text

        except Exception as e:
            error_msg = f"Произошла ошибка при обработке запроса: {str(e)}"
            print(f"ProductAgent error: {error_msg}")
            print(f"Exception type: {type(e).__name__}")
            import traceback

            traceback.print_exc()
            return error_msg

    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        """Собирает промпт для модели.

        Args:
            user_input: Входной запрос пользователя
            **kwargs: Дополнительные параметры

        Returns:
            Строка с промптом
        """
        # Промпт формируется в _create_agent_executor, здесь возвращаем базовый
        return user_input

    def _create_tools(self) -> List[Any]:
        """Создаёт и возвращает список инструментов.

        Returns:
            Список инструментов агента
        """
        return self.tools
