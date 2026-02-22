# Промпты и бизнес-логика

## Источники промптов

Промпты хранятся в **Langfuse** и загружаются по имени. Лейбл задаётся через `LANGFUSE_PROMPT_LABEL` (по умолчанию `production`). TTL кэша: 600 секунд.

| Файл | Описание |
|------|----------|
| `src/services/langfuse/prompt_service.py` | LangfusePromptService — загрузка и компиляция |
| `src/services/ai/prompt.py` | `get_prompt`, `compose_prompts`, кэширование |
| `src/constants.py` | Имена промптов (Langfuse) |

---

## Перечень промптов (Langfuse)

| Константа | Имя в Langfuse | Назначение |
|-----------|----------------|------------|
| `PROMPT_NAME_SYSTEM_PROMPT` | Системный промт | Базовое описание роли ассистента |
| `PROMPT_NAME_PROFILE` | Профиль | Контекст профиля клиента (client_phone) |
| `PROMPT_NAME_PRODUCTS` | Товары | Инфо о товарах для init |
| `PROMPT_NAME_OFFER` | Предложение | Шаблон предложения товаров |
| `PROMPT_NAME_REFLECTOR` | Рефлектор | Рефлексия/итог для init |
| `PROMPT_NAME_COORDINATOR` | Координатор | Координация для process |
| `PROMPT_NAME_FUNCTION` | function | Вызов инструментов |
| `PROMPT_NAME_INFO` | info | Информационные ответы |
| `PROMPT_NAME_UNCLEAR` | unclear | Обработка неясных запросов |
| `PROMPT_NAME_HUMAN_IN_THE_LOOP` | Позвать человека (HITL) | Сообщение при ошибке/пустом ответе |
| `PROMPT_NAME_STYLE_EDUARD` | Стиль Эдуард | Стиль общения |
| `PROMPT_NAME_STYLE_POLINA` | Стиль Полина | Стиль общения |
| `PROMPT_NAME_STYLE_MASHA` | Стиль Маша | Стиль общения |
| `PROMPT_NAME_ERROR_HANDLER` | Обработчик ошибок | Обработка ошибок |

---

## Композиция промптов

### initConversation

```
Системный промт
↓
Профиль (client_phone)
↓
Товары
↓
Предложение
↓
[Стиль: Эдуард | Полина | Маша] — опционально, из clients.style
↓
Рефлектор
```

### processConversation

```
Системный промт
↓
Профиль (client_phone)
↓
Координатор
↓
function
↓
info
↓
unclear
```

---

## Бизнес-логика валидации

### CustomerService (`src/services/ai/customer.py`)

| Метод | Условие | Ошибка |
|-------|---------|--------|
| `validate_client_exists` | client в БД | "Client not found in database" |
| `validate_message_sending_enabled` | `send_message = true` | "Message sending disabled" |
| `validate_conversation_initialized` | `history_count > 0` | "Conversation not initialized" |
| `validate_client_for_conversation` | Все три проверки | Объединённая валидация для process |

**processConversation**: клиент должен существовать, иметь `send_message=true` и хотя бы одно сообщение в истории (т.е. до этого был вызов `initConversation`).

**initConversation**: проверка только на существование клиента в БД (валидатор в Pydantic).

---

## Выбор стиля общения

Из `clients.style` (БД) получается значение:

| style (БД) | Промпт |
|------------|--------|
| эдуард | Стиль Эдуард |
| полина | Стиль Полина |
| маша | Стиль Маша |
| другое/пусто | без дополнительного промпта |

---

## Обработка ошибок в агенте

1. **Timeout** (`MAX_AGENT_EXECUTION_TIME = 1800`): возвращается промпт HITL.
2. **Исключение при выполнении**: возвращается промпт HITL.
3. **Пустой/короткий ответ** (< 3 символов): возвращается промпт HITL.

---

## Инструменты агента (Tools)

| Tool | Описание | Когда использовать |
|------|----------|--------------------|
| `get_client_profile` | Профиль клиента | Контакты, стиль, предпочтения |
| `get_client_orders` | Заказы клиента | История заказов |
| `vector_search` | Семантический поиск | Описание товара словами |
| `get_product_by_title` | Поиск по названию | Точное название |
| `get_random_products` | Случайные товары | Без конкретных критериев |
| `get_database_schema` | Схема БД | Для SQL |
| `generate_sql_from_text` | SQL из текста | Числовые критерии (цена, вес) |
| `execute_sql_query` | Выполнение SQL | Фильтрация по данным |
| `show_product_photos` | Фото товаров | Показать фото клиенту |
| `send_pricelist` | Прайс-лист | Отправка Excel |
| `set_photo_requirement` | Флаг «только с фото» | Контекст для поиска |

---

## Системные значения (system)

Из таблицы `system` в БД:

| topic | Использование |
|-------|---------------|
| `Прайс-лист` (SYSTEM_VALUE_PRICELIST) | URL прайс-листа для `send_pricelist` |
| Прочие ключи | `get_all_system_values()` — динамические переменные в промптах |

---

## Параметры LLM

| Параметр | Значение |
|----------|----------|
| Temperature | 0.5 (DEFAULT_TEMPERATURE) |
| Text-to-SQL | 0.1 (TEXT_TO_SQL_TEMPERATURE) |

---

## Middleware агента

1. **ModelRetryMiddleware** — retry при ConnectionError, TimeoutError (2 попытки).
2. **handle_tool_errors** — обработка ошибок инструментов.
3. **save_product_ids_middleware** — сохранение product_ids в контексте.
4. **ToolRetryMiddleware** — retry для tools (3 попытки).
5. **ModelCallLimitMiddleware** — лимит итераций LLM (`MAX_AGENT_ITERATIONS`).

