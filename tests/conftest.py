"""Pytest configuration и фикстуры для тестов."""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mock_langchain_agents = MagicMock()
mock_langchain_agents.AgentExecutor = MagicMock()
mock_langchain_agents.create_openai_tools_agent = MagicMock()
mock_langchain_agents.create_react_agent = MagicMock()

mock_langchain_callbacks = MagicMock()
mock_langchain_callbacks.tracers = MagicMock()
mock_langchain_callbacks.tracers.LangChainTracer = MagicMock()

sys.modules["langchain.agents"] = mock_langchain_agents
sys.modules["langchain.callbacks"] = mock_langchain_callbacks
sys.modules["langchain.callbacks.tracers"] = mock_langchain_callbacks.tracers


mock_utils_package = MagicMock()
mock_utils_package.langchain_retrievers = MagicMock()
mock_utils_package.langchain_retrievers.SupabaseVectorRetriever = MagicMock()

mock_utils_package.prompts = MagicMock()
mock_utils_package.prompts.get_prompt = AsyncMock(return_value=None)
mock_utils_package.prompts.get_all_system_values = AsyncMock(return_value={})
mock_utils_package.prompts.build_prompt_with_context = MagicMock()

def create_mock_langfuse_handler(*args, **kwargs):
    """Создает мок-экземпляр LangfuseHandler с нужными атрибутами."""
    mock_instance = MagicMock()
    mock_instance.used_tools = set()
    mock_instance.tool_calls = []
    mock_instance.save_conversation_to_langfuse = MagicMock()
    return mock_instance

mock_langfuse_handler_class = MagicMock()
mock_langfuse_handler_class.LangfuseHandler = MagicMock(side_effect=create_mock_langfuse_handler)
mock_utils_package.langfuse_handler = mock_langfuse_handler_class

if "src.utils" not in sys.modules:
    sys.modules["src.utils"] = mock_utils_package
sys.modules["src.utils.langchain_retrievers"] = mock_utils_package.langchain_retrievers
sys.modules["src.utils.prompts"] = mock_utils_package.prompts
sys.modules["src.utils.langfuse_handler"] = mock_utils_package.langfuse_handler

test_env = {
    "SUPABASE_URL": "http://localhost:54321",
    "SUPABASE_ANON_KEY": "test-key",
    "SUPABASE_SERVICE_KEY": "test-service-key",
    "OPENROUTER_API_KEY": "test-api-key",
    "MODEL_ID": "test-model",
    "LANGFUSE_PUBLIC_KEY": "test-key",
    "LANGFUSE_SECRET_KEY": "test-secret",
    "LANGFUSE_HOST": "http://localhost",
    "ALIBABA_KEY": "test-key",
    "BASE_ALIBABA_URL": "http://localhost",
    "EMBEDDING_MODEL_ID": "test-model",
}
for key, value in test_env.items():
    if key not in os.environ:
        os.environ[key] = value
