# Документация проекта Myaso

## Оглавление

1. [Введение](#введение)
2. [Архитектура](#архитектура)
3. [Основные компоненты](#основные-компоненты)
4. [API агентов](#api-агентов)
5. [Инструменты (Tools)](#инструменты-tools)
6. [Память и история диалогов](#память-и-история-диалогов)
7. [Callbacks и мониторинг](#callbacks-и-мониторинг)
8. [Конфигурация](#конфигурация)
9. [Примеры использования](#примеры-использования)
10. [Расширение проекта](#расширение-проекта)

---

## Введение

Проект Myaso — это система интеллектуального ассистента для работы с каталогом товаров через WhatsApp. Система использует LangChain для создания агентов на базе LLM (Large Language Models), которые помогают пользователям находить товары, получать информацию о ценах и взаимодействовать с каталогом.

### Основные возможности

- **Семантический поиск товаров** — поиск по текстовому описанию с использованием векторного поиска
- **SQL-фильтрация** — фильтрация товаров по числовым параметрам (цена, вес, скидка)
- **Контекстные диалоги** — поддержка истории диалога с клиентом
- **Мониторинг и трейсинг** — интеграция с Langfuse для отслеживания работы агентов
- **WhatsApp интеграция** — отправка сообщений и изображений через WhatsApp API

---

## Архитектура

### Общая схема

```
┌─────────────────┐
│  WhatsApp API   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   FastAPI       │
│   ai_router.py  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AgentFactory   │ ◄─── Singleton для создания агентов
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ProductAgent   │ ◄─── Реализация BaseAgent
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  create_agent   │ ◄─── LangChain API
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Tools:                              │
│  - vector_search                     │
│  - generate_sql_from_text            │
│  - execute_sql_query                 │
│  - get_client_profile                │
│  - get_random_products               │
│  - media_tools                       │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   Supabase      │ ◄─── База данных
│   PostgreSQL    │
└─────────────────┘
```

### Поток выполнения запроса

1. **Получение запроса** — FastAPI endpoint получает запрос от WhatsApp
2. **Создание памяти** — Инициализация `SupabaseConversationMemory` для клиента
3. **Получение агента** — `AgentFactory` создает или возвращает кэшированный `ProductAgent`
4. **Подготовка контекста** — Загрузка истории диалога, промптов из БД, системных переменных
5. **Выполнение агента** — Агент обрабатывает запрос через LangChain `create_agent`
6. **Вызов инструментов** — Агент использует tools для поиска товаров, выполнения SQL и т.д.
7. **Сохранение результата** — Ответ сохраняется в память и отправляется через WhatsApp

---

## Основные компоненты

### 1. BaseAgent

Абстрактный базовый класс для всех агентов в системе.

```python
from src.agents.base_agent import BaseAgent

class BaseAgent(ABC):
    """Абстрактный базовый класс агентов на LangChain."""
    
    @abstractmethod
    def run(self, user_input: str, **kwargs: Any) -> Any:
        """Запускает основной сценарий агента для входной строки."""
    
    @abstractmethod
    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        """Собирает промпт для модели."""
    
    @abstractmethod
    def _create_tools(self) -> List[Any]:
        """Создаёт и возвращает список инструментов."""
```

### 2. ProductAgent

Главный агент системы, обрабатывающий запросы пользователей о товарах.

**Основные возможности:**
- Семантический поиск товаров через векторный поиск
- SQL-фильтрация по параметрам
- Работа с историей диалога
- Поддержка системных промптов из БД
- Кэширование агентов для оптимизации

**Ключевые методы:**

- `run(user_input, client_phone, topic, is_init_message, endpoint_name)` — главный метод обработки запроса
- `_create_agent(tools)` — создание агента через LangChain `create_agent`
- `_get_agent(tools)` — получение агента из кэша или создание нового

### 3. AgentFactory

Singleton-фабрика для создания и переиспользования агентов.

```python
from src.agents.factory import AgentFactory

# Получение экземпляра фабрики
factory = AgentFactory.instance()

# Создание агента с памятью
memory = await SupabaseConversationMemory(client_phone)
agent = factory.create_product_agent(config={"memory": memory})

# Регистрация нового типа агента
factory.register_agent("support", SupportAgent)
```

---

## API агентов

### LangChain `create_agent` API

Проект использует современный LangChain API `create_agent` вместо устаревшего `AgentExecutor`.

#### Основные отличия

**Старый API (не используется):**
```python
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent

agent = create_openai_tools_agent(model, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=15)
result = executor.invoke({"input": user_input, "chat_history": history})
```

**Новый API (используется):**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

middleware = [ModelCallLimitMiddleware(run_limit=15, exit_behavior="end")]
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    middleware=middleware,
)
result = await agent.ainvoke({"messages": messages})
```

#### Формат вызова

**Входные данные:**
```python
messages = []
if chat_history:
    messages.extend(chat_history)  # История из памяти
messages.append(HumanMessage(content=user_input))  # Новый запрос

result = await agent.ainvoke({"messages": messages})
```

**Выходные данные:**
```python
result = {
    "messages": [
        HumanMessage(...),    # Входные сообщения
        AIMessage(...),       # Промежуточные ответы
        ToolMessage(...),     # Результаты вызова инструментов
        AIMessage(...),       # Финальный ответ
    ]
}

# Извлечение финального ответа
messages_result = result.get("messages", [])
for msg in reversed(messages_result):
    if isinstance(msg, AIMessage):
        response_text = msg.content
        break
```

#### Middleware

Проект использует `ModelCallLimitMiddleware` для ограничения количества итераций агента:

```python
from langchain.agents.middleware import ModelCallLimitMiddleware

middleware = []
if MAX_AGENT_ITERATIONS > 0:
    middleware.append(
        ModelCallLimitMiddleware(
            run_limit=MAX_AGENT_ITERATIONS,  # Максимум 15 итераций
            exit_behavior="end",             # Завершить выполнение
        )
    )
```

#### Timeout для выполнения

Для контроля времени выполнения используется `asyncio.wait_for`:

```python
import asyncio

if MAX_AGENT_EXECUTION_TIME > 0:
    result = await asyncio.wait_for(
        agent.ainvoke({"messages": messages}, config=config),
        timeout=MAX_AGENT_EXECUTION_TIME,  # Максимум 60 секунд
    )
else:
    result = await agent.ainvoke({"messages": messages}, config=config)
```

---

## Инструменты (Tools)

Все инструменты создаются с помощью декоратора `@tool` из LangChain.

### 1. vector_search

**Назначение:** Семантический поиск товаров по текстовому запросу (векторный поиск)

**Использование:**
- Текстовые запросы по названию/типу товара
- Поиск по текстовым атрибутам (без чисел)
- Семантический поиск (синонимы, контекст)

**Параметры:**
- `query: str` — текстовый запрос пользователя о товарах
- `require_photo: bool` — если True, возвращает только товары с фотографиями

**Пример:**
```python
result = await vector_search.ainvoke({
    "query": "говядина вырезка",
    "require_photo": True
})
# Возвращает: список товаров с ID в секции [PRODUCT_IDS]
```

### 2. generate_sql_from_text

**Назначение:** Генерирует SQL запрос из текстового описания на русском языке

**Автоматический выбор типа запроса:**
- Простой запрос (только фильтрация по products) → генерирует WHERE условия
- Сложный запрос (нужен JOIN с price_history) → генерирует полный SELECT запрос

**Использование:**
- Числовые условия по ЦЕНЕ
- Числовые условия по ВЕСУ
- Числовые условия по СКИДКЕ
- Комбинации числовых условий
- Запросы с JOIN

**Параметры:**
- `text_conditions: str` — текстовое описание условий на русском языке
- `topic: Optional[str]` — тема диалога для загрузки промпта из БД

**Пример:**
```python
sql = await generate_sql_from_text.ainvoke({
    "text_conditions": "товары дешевле 500 рублей за килограмм"
})
# Возвращает: "order_price_kg < 500" или полный SELECT запрос
```

### 3. execute_sql_query

**Назначение:** Универсальный инструмент для выполнения ЛЮБЫХ SQL SELECT запросов

**Принимает:**
- WHERE условия → автоматически оборачивает в SELECT ... FROM myaso.products WHERE ...
- Полные SELECT запросы → выполняет как есть

**Параметры:**
- `sql_query: str` — SQL запрос (WHERE условия или полный SELECT)
- `limit: int` — максимальное количество товаров (по умолчанию 50)

**Безопасность:**
- Запрещены: DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, TRUNCATE, EXECUTE
- Разрешены: только SELECT запросы

**Пример:**
```python
result = await execute_sql_query.ainvoke({
    "sql_query": "order_price_kg < 500",
    "limit": 20
})
# Возвращает: список товаров с ID в секции [PRODUCT_IDS]
```

### 4. get_client_profile

**Назначение:** Получает профиль клиента по номеру телефона

**Возвращает:**
- Информацию о клиенте из БД
- История заказов
- Статус клиента

### 5. get_random_products

**Назначение:** Fallback инструмент для получения случайных товаров

**Использование только когда:**
- `vector_search` вернул "Товары не найдены"
- `execute_sql_query` вернул "Товары не найдены"
- Нужно показать примеры товаров из ассортимента

**Параметры:**
- `limit: int` — количество товаров (по умолчанию 10)

### 6. Media Tools

Инструменты для работы с медиа (создаются динамически через `create_media_tools`):

- Отправка изображений товаров
- Работа с фотографиями из каталога

---

## Память и история диалогов

### SupabaseConversationMemory

Реализация `BaseChatMessageHistory` для хранения истории диалогов в Supabase.

**Основные методы:**

```python
from src.utils.memory import SupabaseConversationMemory

# Создание и инициализация памяти
memory = SupabaseConversationMemory(client_phone)
await memory.__ainit__(client_phone)  # Асинхронная инициализация

# Добавление сообщений
await memory.add_messages([
    HumanMessage(content="Привет"),
    AIMessage(content="Здравствуйте!"),
])

# Получение истории
history = await memory.get_messages()  # Возвращает List[BaseMessage]

# Загрузка для LangChain (совместимость с ConversationBufferMemory)
memory_vars = await memory.load_memory_variables({}, return_messages=True)
chat_history = memory_vars.get("history", [])

# Очистка истории
await memory.clear()
```

**Хранение в БД:**

Таблица `conversation_history` в схеме `myaso`:
- `client_phone: str` — номер телефона клиента
- `role: str` — роль сообщения ("user", "assistant", "system", "tool")
- `message: str` — текст сообщения
- `created_at: timestamp` — время создания

### Интеграция с агентом

История автоматически загружается и сохраняется в методе `ProductAgent.run()`:

```python
# Загрузка истории
memory_vars = await self.memory.load_memory_variables({}, return_messages=True)
chat_history = memory_vars.get("history", [])

# Использование в агенте
messages = []
if chat_history:
    messages.extend(chat_history)
messages.append(HumanMessage(content=user_input))

# Сохранение после получения ответа
if not is_init_message:
    await self.memory.add_messages([HumanMessage(content=user_input)])
await self.memory.add_messages([AIMessage(content=response_text)])
```

---

## Callbacks и мониторинг

### 1. LangfuseHandler

Интеграция с Langfuse для трейсинга и мониторинга работы агентов.

**Возможности:**
- Трейсинг вызовов LLM
- Отслеживание вызовов инструментов
- Сохранение полной истории диалога
- Метрики производительности

**Использование:**

```python
from src.utils.callbacks.langfuse_callback import LangfuseHandler

langfuse_handler = LangfuseHandler(
    client_phone=client_phone,
    session_id=f"{client_phone}_{date.today()}",
    trace_name="ProductAgent",
)

config = {
    "callbacks": [langfuse_handler],
    "metadata": {
        "phone": client_phone,
        "user_id": client_phone,
        "trace_name": "ProductAgent",
    },
    "run_name": "ProductAgent",
    "tags": ["product_agent", "conversation"],
}

# После выполнения агента
langfuse_handler.save_conversation_to_langfuse()
```

### 2. ReasoningLogger

Логирование процесса рассуждения агента для отладки.

**Метрики:**
- Количество вызовов LLM
- Количество вызовов без инструментов
- Сводка по инструментам

**Использование:**

```python
from src.utils.callbacks.reasoning_logger import ReasoningLogger

reasoning_logger = ReasoningLogger(client_phone=client_phone)

config = {
    "callbacks": [reasoning_logger],
    # ...
}

# После выполнения
summary = reasoning_logger.get_summary()
# {
#     "llm_calls": 5,
#     "llm_calls_without_tools": 2,
#     "tools": {...}
# }
```

### 3. StdOutCallbackHandler

Стандартный callback для вывода в консоль (для отладки).

```python
from langchain_core.callbacks.stdout import StdOutCallbackHandler

stdout_handler = StdOutCallbackHandler()
config = {"callbacks": [stdout_handler]}
```

---

## Конфигурация

### Константы (`src/config/constants.py`)

```python
# Ограничения агента
MAX_AGENT_ITERATIONS = 15          # Максимум итераций агента
MAX_AGENT_EXECUTION_TIME = 60      # Максимум времени выполнения (секунды)

# Настройки температуры
DEFAULT_TEMPERATURE = 0.5          # Температура для основного LLM
TEXT_TO_SQL_TEMPERATURE = 0.1      # Температура для генерации SQL

# Лимиты поиска
DEFAULT_SQL_LIMIT = 50             # Лимит результатов SQL запросов
MAX_SQL_LIMIT = 100                # Максимальный лимит

# Безопасность SQL
DANGEROUS_SQL_KEYWORDS = [
    "DROP", "TRUNCATE", "DELETE", "INSERT", 
    "UPDATE", "ALTER", "CREATE", "EXECUTE", "EXEC"
]
```

### Настройки (`src/config/settings.py`)

Проект использует Pydantic Settings для конфигурации:

- `openrouter` — настройки OpenRouter API (LLM)
- `database` — настройки подключения к PostgreSQL
- `supabase` — настройки Supabase
- `whatsapp` — настройки WhatsApp API

---

## Примеры использования

### Пример 1: Обработка сообщения пользователя

```python
from src.agents.factory import AgentFactory
from src.utils.memory import SupabaseConversationMemory

# Создание памяти
memory = SupabaseConversationMemory(client_phone="+79991234567")
await memory.__ainit__(client_phone)

# Получение агента
factory = AgentFactory.instance()
agent = factory.create_product_agent(config={"memory": memory})

# Обработка запроса
response = await agent.run(
    user_input="Найди говядину дешевле 500 рублей",
    client_phone="+79991234567",
    topic=None,
    is_init_message=False,
    endpoint_name="processConversation",
)

print(response)
```

### Пример 2: Инициализация новой беседы

```python
# Очистка истории
memory = await SupabaseConversationMemory(client_phone)
await memory.__ainit__(client_phone)
await memory.clear()

# Создание агента
agent = factory.create_product_agent(config={"memory": memory})

# Отправка приветственного сообщения
response = await agent.run(
    user_input=welcome_prompt,  # Промпт из БД
    client_phone=client_phone,
    topic="init",
    is_init_message=True,
    endpoint_name="initConversation",
)
```

### Пример 3: Создание нового типа агента

```python
from src.agents.base_agent import BaseAgent

class SupportAgent(BaseAgent):
    """Агент поддержки клиентов."""
    
    async def run(self, user_input: str, **kwargs: Any) -> str:
        # Реализация логики поддержки
        pass
    
    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        # Построение промпта
        pass
    
    def _create_tools(self) -> List[Any]:
        # Список инструментов для поддержки
        return []

# Регистрация нового агента
factory = AgentFactory.instance()
factory.register_agent("support", SupportAgent)

# Использование
agent = factory.get_agent("support", config={...})
```

### Пример 4: Создание пользовательского инструмента

```python
from langchain_core.tools import tool

@tool
async def get_weather(city: str) -> str:
    """Получает погоду в указанном городе.
    
    Args:
        city: Название города
        
    Returns:
        Информация о погоде
    """
    # Логика получения погоды
    return f"Погода в {city}: 20°C, солнечно"

# Использование в агенте
tools = [vector_search, get_weather]
agent = create_agent(model=llm, tools=tools, system_prompt=prompt)
```

---

## Расширение проекта

### Добавление нового агента

1. **Создайте класс агента:**

```python
from src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    async def run(self, user_input: str, **kwargs: Any) -> str:
        # Реализация
        pass
    
    def _build_prompt(self, user_input: str, **kwargs: Any) -> str:
        # Построение промпта
        pass
    
    def _create_tools(self) -> List[Any]:
        # Список инструментов
        return []
```

2. **Зарегистрируйте агента:**

```python
from src.agents.factory import AgentFactory

factory = AgentFactory.instance()
factory.register_agent("custom", CustomAgent)
```

3. **Создайте endpoint (опционально):**

```python
from src.routers.ai_router import router

@router.post("/customEndpoint")
async def custom_endpoint(request: CustomRequest):
    factory = AgentFactory.instance()
    agent = factory.get_agent("custom", config={...})
    response = await agent.run(user_input=request.message, ...)
    return {"response": response}
```

### Добавление нового инструмента

1. **Создайте функцию с декоратором `@tool`:**

```python
from langchain_core.tools import tool

@tool
async def my_custom_tool(param: str) -> str:
    """Описание инструмента.
    
    Args:
        param: Описание параметра
        
    Returns:
        Описание возвращаемого значения
    """
    # Логика инструмента
    return result
```

2. **Добавьте инструмент в агент:**

```python
# В методе ProductAgent.__init__ или _create_tools
self.tools = [
    vector_search,
    get_random_products,
    my_custom_tool,  # Новый инструмент
]
```

### Настройка промптов из БД

Системные промпты хранятся в таблице БД и загружаются динамически:

```python
from src.utils.prompts import get_prompt

# Загрузка промпта по теме
prompt = await get_prompt(topic="Вступительное сообщение")

# Промпт автоматически подставляется в SYSTEM_PROMPT агента
```

### Настройка системных переменных

Системные переменные загружаются из БД и используются в промптах:

```python
from src.utils.prompts import get_all_system_values

system_vars = await get_all_system_values()
# {
#     "Прайс-лист": "https://...",
#     "Название компании": "...",
#     ...
# }
```

---

## Структура проекта

```
myaso/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Базовый класс агентов
│   │   ├── factory.py             # Фабрика агентов
│   │   ├── product_agent.py       # Основной агент
│   │   ├── tools/                 # Инструменты
│   │   │   ├── client_tools.py    # Инструменты для работы с клиентами
│   │   │   ├── media_tools.py     # Инструменты для медиа
│   │   │   ├── product_tools.py   # Инструменты для товаров
│   │   │   └── sql_tools.py       # SQL инструменты
│   │   └── prompts/               # Промпты агентов
│   ├── config/                    # Конфигурация
│   │   ├── constants.py           # Константы
│   │   ├── settings.py            # Настройки
│   │   └── ...
│   ├── database/                  # Работа с БД
│   │   ├── database.py            # Подключение к БД
│   │   └── queries/               # SQL запросы
│   ├── models/                    # Модели данных
│   ├── routers/                   # FastAPI роутеры
│   │   └── ai_router.py           # API для агентов
│   ├── services/                  # Сервисы
│   │   └── whatsapp_service.py    # WhatsApp API
│   └── utils/                     # Утилиты
│       ├── callbacks/             # Callbacks для LangChain
│       ├── memory/                # Память диалогов
│       └── retrievers/            # Векторные ретриверы
├── docs/                          # Документация
├── scripts/                       # Скрипты
└── requirements.txt               # Зависимости
```

---

## Полезные ссылки

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [Langfuse Documentation](https://langfuse.com/docs)

---

**Последнее обновление:** 2025-01-24

