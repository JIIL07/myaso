import pytest
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Any

from agents.product_agent import ProductAgent
from agents.tools import enhance_user_product_query, show_product_photos, get_client_profile
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class TestProductAgentCreation:
    """Тесты создания ProductAgent."""

    @patch("agents.product_agent.langchain_settings")
    @patch("agents.product_agent.ChatOpenAI")
    def test_create_agent_with_defaults(self, mock_chat_openai, mock_langchain_settings):
        """Тест создания агента с параметрами по умолчанию."""
        mock_langchain_settings.langsmith_tracing_enabled = False
        mock_langchain_settings.langsmith_api_key = None
        mock_langchain_settings.setup_langsmith_tracing = Mock()
        
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        agent = ProductAgent()

        assert agent is not None
        assert agent.llm == mock_llm
        assert agent.agent_type == "openai-tools"
        assert len(agent.tools) == 3
        assert enhance_user_product_query in agent.tools
        assert show_product_photos in agent.tools
        assert get_client_profile in agent.tools
        assert agent._agent_executor is None

    @patch("agents.product_agent.langchain_settings")
    @patch("agents.product_agent.ChatOpenAI")
    def test_create_agent_with_custom_llm(self, mock_chat_openai, mock_langchain_settings):
        """Тест создания агента с кастомным LLM."""
        mock_langchain_settings.langsmith_tracing_enabled = False
        mock_langchain_settings.langsmith_api_key = None
        mock_langchain_settings.setup_langsmith_tracing = Mock()
        
        custom_llm = MagicMock()
        
        agent = ProductAgent(llm=custom_llm)

        assert agent.llm == custom_llm
        mock_chat_openai.assert_not_called()

    @patch("agents.product_agent.langchain_settings")
    @patch("agents.product_agent.ChatOpenAI")
    def test_create_agent_with_custom_tools(self, mock_chat_openai, mock_langchain_settings):
        """Тест создания агента с кастомными инструментами."""
        mock_langchain_settings.langsmith_tracing_enabled = False
        mock_langchain_settings.langsmith_api_key = None
        mock_langchain_settings.setup_langsmith_tracing = Mock()
        
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        custom_tools = [MagicMock(), MagicMock()]

        agent = ProductAgent(tools=custom_tools)

        assert agent.tools == custom_tools
        assert len(agent.tools) == 2

    @patch("agents.product_agent.langchain_settings")
    @patch("agents.product_agent.ChatOpenAI")
    def test_create_agent_with_react_type(self, mock_chat_openai, mock_langchain_settings):
        """Тест создания агента с типом zero-shot-react-description."""
        mock_langchain_settings.langsmith_tracing_enabled = False
        mock_langchain_settings.langsmith_api_key = None
        mock_langchain_settings.setup_langsmith_tracing = Mock()
        
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        agent = ProductAgent(agent_type="zero-shot-react-description")

        assert agent.agent_type == "zero-shot-react-description"


class TestProductAgentToolCalling:
    """Тесты вызова инструментов агента."""

    @pytest.fixture
    def mock_agent_executor(self):
        """Фикстура для мокирования AgentExecutor."""
        executor = AsyncMock()
        executor.ainvoke = AsyncMock()
        return executor

    @pytest.fixture
    def agent(self, mock_agent_executor):
        """Фикстура для создания агента с мокированным executor."""
        with patch("agents.product_agent.langchain_settings"), \
             patch("agents.product_agent.ChatOpenAI") as mock_chat_openai, \
             patch("agents.product_agent.create_openai_tools_agent") as mock_create_agent, \
             patch("agents.product_agent.AgentExecutor") as mock_agent_executor_class:
            
            mock_langchain_settings = MagicMock()
            mock_langchain_settings.langsmith_tracing_enabled = False
            mock_langchain_settings.langsmith_api_key = None
            mock_langchain_settings.setup_langsmith_tracing = Mock()
            
            mock_llm = MagicMock()
            mock_chat_openai.return_value = mock_llm
            mock_agent_executor_class.return_value = mock_agent_executor
            
            agent = ProductAgent()
            agent._agent_executor = mock_agent_executor
            yield agent

    @pytest.mark.asyncio
    async def test_run_without_memory(self, agent, mock_agent_executor):
        """Тест выполнения агента без памяти."""
        mock_agent_executor.ainvoke.return_value = {
            "output": "Извините, товары не найдены."
        }

        with patch("agents.product_agent.get_client_profile") as mock_get_profile:
            mock_get_profile.ainvoke = AsyncMock(return_value="Профиль клиента не найден в базе данных.")
            
            result = await agent.run("Покажи мне стейки", "+1234567890")

        assert result == "Извините, товары не найдены."
        mock_agent_executor.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_memory(self, agent, mock_agent_executor):
        """Тест выполнения агента с памятью."""
        mock_memory = AsyncMock()
        mock_memory.load_memory_variables = AsyncMock(
            return_value={"history": [HumanMessage(content="Привет"), AIMessage(content="Здравствуйте!")]}
        )
        mock_memory.add_messages = AsyncMock()
        agent.memory = mock_memory

        mock_agent_executor.ainvoke.return_value = {
            "output": "Вот список стейков."
        }

        with patch("agents.product_agent.get_client_profile") as mock_get_profile:
            mock_get_profile.ainvoke = AsyncMock(return_value="Профиль клиента не найден в базе данных.")
            
            result = await agent.run("Покажи стейки", "+1234567890")

        assert result == "Вот список стейков."
        mock_memory.load_memory_variables.assert_called_once()
        mock_memory.add_messages.assert_called()

    @pytest.mark.asyncio
    async def test_run_with_client_profile(self, agent, mock_agent_executor):
        """Тест выполнения агента с профилем клиента."""
        mock_agent_executor.ainvoke.return_value = {
            "output": "Учитывая ваш профиль, вот подходящие товары."
        }

        with patch("agents.product_agent.get_client_profile") as mock_get_profile:
            mock_get_profile.ainvoke = AsyncMock(return_value="Имя: Иван\nГород: Москва")
            
            result = await agent.run("Что у вас есть?", "+1234567890")

        assert result == "Учитывая ваш профиль, вот подходящие товары."
        mock_agent_executor.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_error_handling(self, agent, mock_agent_executor):
        """Тест обработки ошибок при выполнении агента."""
        mock_agent_executor.ainvoke.side_effect = Exception("Ошибка выполнения")

        with patch("agents.product_agent.get_client_profile") as mock_get_profile:
            mock_get_profile.ainvoke = AsyncMock(return_value="Профиль клиента не найден в базе данных.")
            
            result = await agent.run("Тест", "+1234567890")

        assert "пошло не так" in result.lower() or "напишите еще раз" in result.lower()


class TestProductAgentSimpleRequest:
    """Тесты обработки простых запросов."""

    @pytest.fixture
    def agent(self):
        """Фикстура для создания агента."""
        with patch("agents.product_agent.langchain_settings"), \
             patch("agents.product_agent.ChatOpenAI") as mock_chat_openai, \
             patch("agents.product_agent.create_openai_tools_agent"), \
             patch("agents.product_agent.AgentExecutor") as mock_agent_executor_class:
            
            mock_langchain_settings = MagicMock()
            mock_langchain_settings.langsmith_tracing_enabled = False
            mock_langchain_settings.langsmith_api_key = None
            mock_langchain_settings.setup_langsmith_tracing = Mock()
            
            mock_llm = MagicMock()
            mock_chat_openai.return_value = mock_llm
            
            executor = AsyncMock()
            executor.ainvoke = AsyncMock()
            mock_agent_executor_class.return_value = executor
            
            agent = ProductAgent()
            agent._agent_executor = executor
            yield agent

    @pytest.mark.asyncio
    async def test_simple_greeting(self, agent):
        """Тест обработки простого приветствия."""
        agent._agent_executor.ainvoke.return_value = {
            "output": "Здравствуйте! Чем могу помочь?"
        }

        with patch("agents.product_agent.get_client_profile") as mock_get_profile:
            mock_get_profile.ainvoke = AsyncMock(return_value="Профиль клиента не найден в базе данных.")
            
            result = await agent.run("Привет", "+1234567890")

        assert "Здравствуйте" in result or "помочь" in result.lower()

    @pytest.mark.asyncio
    async def test_product_search_request(self, agent):
        """Тест запроса поиска товаров."""
        agent._agent_executor.ainvoke.return_value = {
            "output": "Найдено товаров: 3\n\nСтейк Рибай\nЦена: 500 руб/кг"
        }

        with patch("agents.product_agent.get_client_profile") as mock_get_profile:
            mock_get_profile.ainvoke = AsyncMock(return_value="Профиль клиента не найден в базе данных.")

            result = await agent.run("Покажи стейки", "+1234567890")

        assert "Найдено" in result or "Стейк" in result

    @pytest.mark.asyncio
    async def test_photo_request(self, agent):
        """Тест запроса фотографий товаров."""
        agent._agent_executor.ainvoke.return_value = {
            "output": "Фотографии отправлены"
        }

        with patch("agents.product_agent.get_client_profile") as mock_get_profile:
            mock_get_profile.ainvoke = AsyncMock(return_value="Профиль клиента не найден в базе данных.")
            
            result = await agent.run("Покажи фото стейков", "+1234567890")

        assert len(result) > 0


class TestProductAgentToolsMocking:
    """Тесты мокирования инструментов Supabase."""

    @pytest.mark.asyncio
    async def test_enhance_user_product_query_mock(self):
        """Тест мокирования enhance_user_product_query."""
        with patch("agents.tools.SupabaseVectorRetriever") as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever_class.return_value = mock_retriever
            
            mock_documents = [
                Document(
                    page_content="Title: Стейк Рибай; Price/kg: 500",
                    metadata={
                        "title": "Стейк Рибай",
                        "supplier_name": "Поставщик 1",
                        "order_price_kg": 500,
                        "min_order_weight_kg": 5,
                    }
                )
            ]
            mock_retriever.get_relevant_documents = AsyncMock(return_value=mock_documents)

            result = await enhance_user_product_query.ainvoke({"query": "стейки"})

            assert "Найдено товаров" in result or "Стейк Рибай" in result
            mock_retriever.get_relevant_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_profile_mock(self):
        """Тест мокирования get_client_profile."""
        mock_profile_data = {
            "name": "Иван",
            "phone": "+1234567890",
            "city": "Москва",
            "business_area": "Ресторан"
        }
        
        with patch("agents.tools.acreate_client") as mock_create_client:
            mock_eq = MagicMock()
            mock_select = MagicMock()
            mock_table = MagicMock()
            
            mock_execute_result = MagicMock()
            mock_execute_result.data = [mock_profile_data]
            mock_eq.execute = AsyncMock(return_value=mock_execute_result)
            
            mock_table.select.return_value = mock_select
            mock_select.eq.return_value = mock_eq
            
            mock_supabase = AsyncMock()
            mock_supabase.table = MagicMock(return_value=mock_table)
            
            mock_create_client.return_value = mock_supabase
            
            result = await get_client_profile.ainvoke({"phone": "+1234567890"})

            assert "Иван" in result or "Москва" in result or "Ресторан" in result

    @pytest.mark.asyncio
    async def test_show_product_photos_mock(self):
        """Тест мокирования show_product_photos."""
        with patch("agents.tools.supabase_client") as mock_supabase, \
             patch("agents.tools.httpx.AsyncClient") as mock_httpx_client:
            
            mock_table = MagicMock()
            mock_select = MagicMock()
            mock_eq = MagicMock()
            
            mock_supabase.table.return_value = mock_table
            mock_table.select.return_value = mock_select
            mock_select.eq.return_value = mock_eq
            mock_eq.execute.return_value = MagicMock(data=[
                {"title": "Стейк Рибай", "photo": "http://example.com/photo.jpg"}
            ])
            
            mock_client_instance = AsyncMock()
            mock_client_instance.post = AsyncMock()
            mock_httpx_client.return_value.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_httpx_client.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await show_product_photos.ainvoke({
                "product_titles": ["Стейк Рибай"],
                "phone": "+1234567890"
            })

            assert len(result) > 0
            mock_client_instance.post.assert_called()