# Миграция на LangChain: Документация по архитектуре агентов

## Обзор

Этот документ описывает новую архитектуру агентов на основе LangChain, которая заменила предыдущий подход с использованием Mirascope. Архитектура обеспечивает гибкость, расширяемость и лучшую интеграцию с экосистемой LangChain.

## Архитектура

### Компоненты системы

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Router                          │
│                  (ai_router.py)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    AgentFactory                              │
│              (Singleton, кэширование)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    ProductAgent                              │
│              (BaseAgent implementation)                      │
├─────────────────────────────────────────────────────────────┤
│  - AgentExecutor (LangChain)                                 │
│  - LLM (ChatOpenAI)                                          │
│  - Tools (LangChain Tools)                                   │
│  - Memory (SupabaseConversationMemory)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Tools     │ │   Memory    │ │   Retriever │
│             │ │             │ │             │
│ - enhance   │ │ Supabase    │ │ Vector      │
│ - photos    │ │ History     │ │ Search      │
│ - profile   │ │             │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
```

### Базовые классы

#### BaseAgent

Абстрактный базовый класс для всех агентов:

```python
class BaseAgent(ABC):
    def __init__(self, *, model: Any, tools: List[Any], config: Dict[str, Any]):
        self.model = model
        self.tools = tools
        self.config = config
    
    @abstractmethod
    async def run(self, user_input: str, **kwargs) -> Any:
        """Запускает основной сценарий агента"""
    
    @abstractmethod
    def _build_prompt(self, user_input: str, **kwargs) -> str:
        """Собирает промпт для модели"""
    
    @abstractmethod
    def _create_tools(self) -> List[Any]:
        """Создаёт список инструментов"""
```

#### ProductAgent

Реализация агента для работы с продуктами:

- **AgentExecutor**: Использует LangChain AgentExecutor для управления выполнением агента
- **Tools**: Инструменты для поиска товаров, отправки фото, получения профиля
- **Memory**: Интеграция с Supabase для хранения истории диалога
- **Retriever**: Векторный поиск по товарам (опционально)

### Инструменты (Tools)

Инструменты определены как асинхронные функции с декоратором `@tool`:

```python
from langchain_core.tools import tool

@tool
async def enhance_user_product_query(query: str) -> str:
    """Ищет товары в базе данных по семантическому запросу."""
    # Реализация векторного поиска
    ...
```

Основные инструменты:
- `enhance_user_product_query`: Семантический поиск товаров
- `show_product_photos`: Отправка фотографий через WhatsApp API
- `get_client_profile`: Получение профиля клиента

### Память (Memory)

Интеграция с Supabase для хранения истории диалога:

```python
from src.utils.langchain_memory import SupabaseConversationMemory

memory = SupabaseConversationMemory(client_phone="+1234567890")
await memory.__ainit__()
```

## Как добавить нового агента

### Шаг 1: Создание класса агента

Создайте новый файл в `agents/`, например `support_agent.py`:

```python
from agents.base_agent import BaseAgent
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from src.config.settings import settings

class SupportAgent(BaseAgent):
    """Агент для обработки вопросов поддержки."""
    
    def __init__(self, *, llm=None, tools=None, **kwargs):
        if llm is None:
            llm = ChatOpenAI(
                model=settings.openrouter.model_id,
                openai_api_key=settings.openrouter.openrouter_api_key,
                openai_api_base=settings.openrouter.base_url,
            )
        
        if tools is None:
            tools = [your_custom_tool1, your_custom_tool2]
        
        super().__init__(model=llm, tools=tools, config=kwargs)
        self.llm = llm
        self._agent_executor = None
    
    def _create_agent_executor(self) -> AgentExecutor:
        system_prompt = "Ты помощник службы поддержки..."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )
    
    async def run(self, user_input: str, client_phone: str) -> str:
        if self._agent_executor is None:
            self._agent_executor = self._create_agent_executor()
        
        result = await self._agent_executor.ainvoke({
            "input": user_input,
            "chat_history": [],
        })
        
        return result.get("output", "")
    
    def _build_prompt(self, user_input: str, **kwargs) -> str:
        return user_input
    
    def _create_tools(self) -> List[Any]:
        return self.tools
```

### Шаг 2: Регистрация агента в фабрике

В `agents/factory.py`:

```python
from .support_agent import SupportAgent

class AgentFactory:
    def __init__(self):
        self.registered_agents = {}
        self.register_agent("product", ProductAgent)
        self.register_agent("support", SupportAgent)  # Добавьте это
```

### Шаг 3: Создание инструментов

В `agents/tools.py` или отдельном файле:

```python
from langchain_core.tools import tool

@tool
async def get_support_ticket(phone: str) -> str:
    """Получает информацию о тикете поддержки."""
    # Ваша реализация
    ...
```

### Шаг 4: Использование в роутере

В `src/routers/ai_router.py`:

```python
factory = AgentFactory.instance()
agent = factory.create_product_agent(config={})
response = await agent.run(user_input=request.message, client_phone=request.client_phone)
```

## API Endpoints

После завершения миграции доступны следующие endpoints:

- `POST /ai/initConversation` - Инициализация нового диалога
- `POST /ai/processConversation` - Обработка сообщения пользователя
- `GET /ai/getProfile` - Получение профиля клиента
- `DELETE /ai/resetConversation` - Сброс истории диалога

### Миграция инструментов

**Mirascope:**
```python
class ShowProductPhotos(BaseTool):
    products: List[Product] = Field(...)
    phone_number: str = Field(...)
    
    def call(self) -> str:
        # Реализация
        ...
```

**LangChain:**
```python
@tool
async def show_product_photos(
    product_titles: List[str], 
    phone: str
) -> str:
    """Отправляет фотографии товаров."""
    # Реализация
    ...
```

### Миграция агентов

**Mirascope:**
```python
@openai.call(
    model=settings.openrouter.model_id,
    tools=[ShowProductPhotos, EnhanceUserProductQuery],
)
class LLMService:
    async def infer(self, query: str, history: List):
        # Прямой вызов модели
        ...
```

**LangChain:**
```python
class ProductAgent(BaseAgent):
    def __init__(self, ...):
        # Настройка LLM, tools, memory
        ...
    
    async def run(self, user_input: str, client_phone: str):
        # Использование AgentExecutor
        result = await self._agent_executor.ainvoke(...)
        ...
```

## Тестирование

См. `tests/test_product_agent.py` для примеров тестирования агентов:

- Мокирование внешних зависимостей (Supabase, LLM)
- Тестирование создания агентов
- Тестирование вызова инструментов
- Тестирование обработки запросов


