# Диаграмма пайплайна клиента

## Обзор

Система Myaso — AI-ассистент для магазина мясной продукции. Клиенты общаются через Telegram/WhatsApp, запросы поступают в API, обрабатываются агентом и ответы доставляются обратно.

## Точки входа (API)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ВХОДЯЩИЕ ЗАПРОСЫ                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  POST /get_message          │  Внешний webhook (WhatsApp/Telegram)          │
│  { phone, message }         │  → Валидация телефона → /ai/processConversation│
├─────────────────────────────────────────────────────────────────────────────┤
│  POST /ai/processConversation│  Обработка сообщения клиента                   │
│  { client_phone, message }   │  (требует initConversation)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  POST /ai/initConversation   │  Инициализация новой беседы                   │
│  { client_phone }            │  Первое приветствие с предложением товаров     │
├─────────────────────────────────────────────────────────────────────────────┤
│  DELETE /ai/resetConversation │  Сброс истории диалога                        │
│  { client_phone }            │  Очистка памяти для нового контекста          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Диаграмма пайплайна (Mermaid)

```mermaid
flowchart TB
    subgraph Entry["Точки входа"]
        A1["/get_message\n(phone, message)"]
        A2["/ai/processConversation\n(client_phone, message)"]
        A3["/ai/initConversation\n(client_phone)"]
    end

    subgraph Validation["Валидация"]
        V1[" normalize_phone\n validate_phone"]
        V2["CustomerService.validate_client_for_conversation"]
        V3["• client существует в БД\n• send_message = true\n• история инициализирована"]
    end

    subgraph Conversation["ConversationService"]
        C1["process_conversation / init_conversation"]
        C2{"RateLimiter\nсвободен?"}
        C3["_process_conversation_internal\n/ _init_conversation_internal"]
        C4["QueueManager.add_task\ntype: process | init"]
    end

    subgraph AgentQueue["Agent Queue Worker"]
        Q1["get_task()"]
        Q2["RateLimiter.acquire"]
        Q3["_handle_process_task\n/ _handle_init_task"]
    end

    subgraph Agent["ProductAgent"]
        AG1["_create_agent_with_memory"]
        AG2["_build_*_prompt\n(Langfuse)"]
        AG3["agent.run(user_input, system_prompt)"]
        AG4["LangChain Agent\n+ Tools"]
    end

    subgraph PostProcess["После обработки"]
        P1["_schedule_delayed_reply"]
        P2["send_delayed_message\n→ PGMQ (delay ~15 мин)"]
        P3["Queue Worker\n(PGMQ)"]
    end

    subgraph Delivery["Доставка клиенту"]
        D1["Telegram / WhatsApp API"]
    end

    A1 --> V1
    V1 --> A2
    A2 --> V2
    A3 --> V2
    V2 --> C1
    V3 -.-> V2

    C1 --> C2
    C2 -->|да| C3
    C2 -->|нет| C4
    C4 --> Q1

    Q1 --> Q2
    Q2 --> Q3
    Q3 --> AG1

    AG1 --> AG2
    AG2 --> AG3
    AG3 --> AG4

    AG4 --> P1
    P1 --> P2
    P2 --> P3
    P3 -.-> D1
```

## Последовательность обработки сообщения

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Customer
    participant Conv as ConversationService
    participant Queue as AgentQueue
    participant Agent as ProductAgent
    participant Memory
    participant PGMQ
    participant Messenger

    Client->>API: POST /ai/processConversation
    API->>Customer: validate_client_for_conversation
    Customer->>Customer: client exists? send_message? history?
    alt validation failed
        Customer-->>API: ClientValidationError
        API-->>Client: ErrorResponse
    end

    API->>Conv: process_conversation
    Conv->>Queue: is_available?

    alt Agent available
        Conv->>Memory: SupabaseConversationMemory
        Conv->>Agent: run(message, system_prompt)
        Agent->>Memory: load history
        Agent->>Agent: LLM + Tools
        Agent->>Memory: save (user, assistant)
        Agent-->>Conv: response_text
        Conv->>PGMQ: send_delayed_message(response, delay)
        Conv-->>API: { success: true }
    else Agent busy
        Conv->>Queue: add_task(process)
        Conv-->>API: { success: true, queued: true }
        Queue->>Conv: _handle_process_task (async)
        Note over Queue,Agent: ... same flow ...
    end

    API-->>Client: SuccessResponse

    Note over PGMQ,Messenger: После истечения delay (до 15 мин)
    PGMQ->>Messenger: Queue Worker читает сообщение
    Messenger->>Client: Отправка через Telegram/WhatsApp
```

## Компоненты

| Компонент | Описание |
|-----------|----------|
| **CustomerService** | Проверка клиента: есть в БД, send_message включён, есть история |
| **ConversationService** | Оркестрация: очередь задач, создание агента, построение промптов |
| **AgentQueueWorker** | Обработка задач из in-memory очереди, rate limit (1 concurrent) |
| **ProductAgent** | LangChain-агент с инструментами, LLM (OpenRouter), память |
| **PGMQ** | Очередь сообщений (PostgreSQL) — хранение ответов агента для отложенной доставки (~15 мин). Ответы попадают в очередь через `send_delayed_message` после генерации агентом.