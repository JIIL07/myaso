-- Создание таблицы rules для хранения всех технических правил системы
-- Миграция всех правил из констант Python в базу данных

-- Создание типа для категорий правил
DO $$ BEGIN
    CREATE TYPE myaso.rule_type AS ENUM (
        'limit',           -- Числовые лимиты (MAX_HISTORY_MESSAGES, MAX_SQL_LIMIT и т.д.)
        'timeout',         -- Таймауты (HTTP_TIMEOUT_SECONDS, DB_CONNECTION_TIMEOUT и т.д.)
        'constant',        -- Константы (DEFAULT_TEMPERATURE, DEFAULT_FIELD_VALUE и т.д.)
        'list',            -- Списки (DANGEROUS_SQL_KEYWORDS, GREETING_WORDS, EMPTY_VALUES)
        'boolean',         -- Булевы флаги (ENABLE_QUERY_REWRITING, ENABLE_RERANKING)
        'pattern',         -- Паттерны (phone validation pattern)
        'validation',      -- Правила валидации
        'instruction'      -- Инструкции для агента (SQL Generation Rules, Tool Usage Guidelines и т.д.)
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Создание таблицы rules
CREATE TABLE IF NOT EXISTS myaso.rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(255) NOT NULL UNIQUE,
    rule_type myaso.rule_type NOT NULL,
    rule_value TEXT NOT NULL,  -- JSON для списков и сложных значений, текст для простых
    description TEXT,
    category VARCHAR(100) NOT NULL,  -- 'agent', 'database', 'queue', 'validation', 'search', 'llm', 'general'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Создание индексов для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_rules_category ON myaso.rules(category);
CREATE INDEX IF NOT EXISTS idx_rules_rule_type ON myaso.rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_rules_rule_name ON myaso.rules(rule_name);

-- Комментарии к таблице и колонкам
COMMENT ON TABLE myaso.rules IS 'Таблица для хранения всех технических правил и констант системы';
COMMENT ON COLUMN myaso.rules.rule_name IS 'Уникальное имя правила (соответствует константе в Python)';
COMMENT ON COLUMN myaso.rules.rule_type IS 'Тип правила: limit, timeout, constant, list, boolean, pattern, validation, instruction';
COMMENT ON COLUMN myaso.rules.rule_value IS 'Значение правила (JSON для списков, текст для простых значений)';
COMMENT ON COLUMN myaso.rules.description IS 'Описание правила и его назначения';
COMMENT ON COLUMN myaso.rules.category IS 'Категория правила для группировки';

-- Функция для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION myaso.update_rules_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Триггер для автоматического обновления updated_at
DROP TRIGGER IF EXISTS trigger_update_rules_updated_at ON myaso.rules;
CREATE TRIGGER trigger_update_rules_updated_at
    BEFORE UPDATE ON myaso.rules
    FOR EACH ROW
    EXECUTE FUNCTION myaso.update_rules_updated_at();

-- Предоставление прав доступа
GRANT ALL ON TABLE myaso.rules TO anon, authenticated, service_role;
GRANT USAGE, SELECT ON SEQUENCE myaso.rules_id_seq TO anon, authenticated, service_role;

-- ============================================================================
-- МИГРАЦИЯ ПРАВИЛ ИЗ КОНСТАНТ
-- ============================================================================

-- Правила для агента (Agent Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MAX_HISTORY_MESSAGES', 'limit', '10', 'Максимальное количество сообщений истории для контекста агента', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MAX_AGENT_ITERATIONS', 'limit', '1000', 'Максимальное количество итераций выполнения агента', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MAX_AGENT_EXECUTION_TIME', 'timeout', '3600', 'Максимальное время выполнения агента в секундах', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('AGENT_RECURSION_LIMIT', 'limit', '1005', 'Лимит рекурсии для агента (MAX_AGENT_ITERATIONS + 5)', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для поиска (Search Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('VECTOR_SEARCH_LIMIT', 'limit', '50', 'Максимальный лимит результатов векторного поиска', 'search')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DEFAULT_VECTOR_SEARCH_K', 'limit', '10', 'Количество результатов векторного поиска по умолчанию', 'search')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MAX_HISTORY_FOR_REWRITING', 'limit', '10', 'Максимальное количество сообщений истории для переписывания запросов', 'search')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для SQL (SQL Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DEFAULT_SQL_LIMIT', 'limit', '50', 'Лимит результатов SQL запросов по умолчанию', 'database')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MAX_SQL_LIMIT', 'limit', '100', 'Максимальный лимит результатов SQL запросов', 'database')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MAX_SQL_RETRY_ATTEMPTS', 'limit', '3', 'Максимальное количество попыток повтора SQL запроса', 'database')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DANGEROUS_SQL_KEYWORDS', 'list', '["DROP", "TRUNCATE", "DELETE", "INSERT", "EXECUTE", "EXEC", "UPDATE", "ALTER", "CREATE"]', 'Список опасных SQL ключевых слов, которые запрещены в запросах', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для LLM (LLM Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DEFAULT_TEMPERATURE', 'constant', '0.5', 'Температура LLM по умолчанию', 'llm')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('TEXT_TO_SQL_TEMPERATURE', 'constant', '0.1', 'Температура LLM для генерации SQL запросов', 'llm')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для таймаутов (Timeout Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('HTTP_TIMEOUT_SECONDS', 'timeout', '10.0', 'Таймаут HTTP запросов в секундах', 'general')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DB_CONNECTION_TIMEOUT', 'timeout', '10.0', 'Таймаут подключения к базе данных в секундах', 'database')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DB_COMMAND_TIMEOUT', 'timeout', '30.0', 'Таймаут выполнения команд базы данных в секундах', 'database')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DB_POOL_MIN_SIZE', 'limit', '5', 'Минимальный размер пула соединений с базой данных', 'database')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DB_POOL_MAX_SIZE', 'limit', '20', 'Максимальный размер пула соединений с базой данных', 'database')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для эмбеддингов (Embedding Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('EMBEDDING_DELAY_SECONDS', 'timeout', '0.1', 'Задержка между запросами эмбеддингов в секундах', 'search')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('EMBEDDING_BATCH_SIZE', 'limit', '10', 'Размер батча для обработки эмбеддингов', 'search')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для очереди (Queue Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('QUEUE_CHECK_INTERVAL', 'timeout', '30', 'Интервал проверки очереди в секундах', 'queue')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('VISIBILITY_TIMEOUT', 'timeout', '60', 'Время видимости сообщения при чтении из очереди в секундах', 'queue')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('QUEUE_BATCH_SIZE', 'limit', '10', 'Количество сообщений для обработки из очереди за раз', 'queue')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('QUEUE_DELAY_SECONDS', 'timeout', '900', 'Задержка отправки сообщения в очереди по умолчанию (15 минут)', 'queue')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('QUEUE_NAME', 'constant', 'delayed_messages', 'Имя очереди PGMQ для отложенных сообщений', 'queue')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Булевы флаги (Boolean Flags)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('ENABLE_QUERY_REWRITING', 'boolean', 'false', 'Включить переписывание запросов', 'search')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('ENABLE_RERANKING', 'boolean', 'false', 'Включить реранкинг результатов поиска', 'search')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для кеширования (Cache Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('CONTEXT_CACHE_TTL_SECONDS', 'timeout', '300', 'Время жизни кеша контекста в секундах', 'general')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для валидации и нормализации (Validation Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('DEFAULT_FIELD_VALUE', 'constant', 'по запросу', 'Значение по умолчанию для пустых полей', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('EMPTY_VALUES', 'list', '["не указано", "null", "none", ""]', 'Список значений, которые считаются пустыми', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('GREETING_WORDS', 'list', '["привет", "здравствуй", "здравствуйте", "добрый день", "добрый вечер", "доброе утро", "доброй ночи", "доброго дня", "доброго вечера", "доброго утра", "здорово", "салют", "хай", "hi", "hello", "доброго времени суток", "приветствую", "добро пожаловать"]', 'Список слов приветствия для определения начала разговора', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила валидации телефона (Phone Validation Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('PHONE_VALIDATION_PATTERN', 'pattern', '^\\+[1-9]\\d{9,14}$', 'Регулярное выражение для валидации номера телефона', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('PHONE_MIN_LENGTH', 'limit', '10', 'Минимальная длина номера телефона (количество цифр)', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('PHONE_MAX_LENGTH', 'limit', '15', 'Максимальная длина номера телефона (количество цифр)', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила валидации сообщений (Message Validation Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MESSAGE_MIN_LENGTH', 'limit', '1', 'Минимальная длина сообщения пользователя', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('MESSAGE_MAX_LENGTH', 'limit', '2000', 'Максимальная длина сообщения пользователя', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('TOPIC_MIN_LENGTH', 'limit', '1', 'Минимальная длина темы беседы', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('TOPIC_MAX_LENGTH', 'limit', '100', 'Максимальная длина темы беседы', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('CLIENT_PHONE_MIN_LENGTH', 'limit', '1', 'Минимальная длина номера телефона клиента', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('CLIENT_PHONE_MAX_LENGTH', 'limit', '20', 'Максимальная длина номера телефона клиента', 'validation')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для продуктов (Product Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('RANDOM_PRODUCTS_MAX_LIMIT', 'limit', '20', 'Максимальный лимит случайных товаров', 'general')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('STATISTICS_MAX_LIMIT', 'limit', '20', 'Максимальный лимит результатов статистики товаров', 'general')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('RECOMMENDATIONS_MAX_LIMIT', 'limit', '20', 'Максимальный лимит рекомендаций товаров', 'general')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Правила для LangFuse (LangFuse Rules)
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('LANGFUSE_FLUSH_INTERVAL', 'timeout', '1', 'Интервал отправки событий в LangFuse в секундах', 'llm')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- ИНСТРУКЦИИ ДЛЯ АГЕНТА (Agent Instructions)
-- ============================================================================

-- SQL Generation Rules - правила генерации SQL запросов
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('SQL_GENERATION_RULES', 'instruction', 
    'ПРАВИЛА ГЕНЕРАЦИИ SQL:

1. ВЫБОР ТИПА ЗАПРОСА:
   - Если запрос простой (только фильтрация по таблице products) -> генерируй ТОЛЬКО WHERE условия (без SELECT/FROM)
   - Если нужен JOIN с price_history или сложные подзапросы -> генерируй ПОЛНЫЙ SELECT запрос

2. ДЛЯ WHERE УСЛОВИЙ (простой запрос):
   - Генерируй ТОЛЬКО условия, БЕЗ SELECT/FROM/WHERE
   - Используй ТОЛЬКО колонки из таблицы products
   - Пример: order_price_kg > 100 AND supplier_name = ''Поставщик''

3. ДЛЯ ПОЛНОГО SELECT ЗАПРОСА (сложный запрос с JOIN/подзапросами):
   - Генерируй ПОЛНЫЙ SELECT запрос: SELECT ... FROM myaso.products JOIN myaso.price_history ...
   - Явно указывай схему myaso: myaso.products, myaso.price_history
   - Запрос должен возвращать колонки из myaso.products (обязательно id)
   - ВАЖНО: При JOIN с price_history ВСЕГДА используй DISTINCT или EXISTS, так как в price_history может быть несколько записей для одного товара

4. ОБЩИЕ ПРАВИЛА:
   - Используй ТОЛЬКО колонки из схемы таблиц! Никаких других колонок не существует!
   - НЕ используй алиасы таблиц (p, ph, t и т.д.)
   - НЕ используй ключевое слово AS для алиасов
   - Всегда проверяй наличие колонки в схеме перед использованием
   - Для числовых сравнений используй правильные типы данных
   - Для текстовых поисков ВСЕГДА используй ILIKE (регистронезависимый поиск) с правильным экранированием
   - НИКОГДА не используй LIKE для текстовых поисков, только ILIKE
   - Для сравнения текстовых полей используй ILIKE вместо = или LIKE',
    'Правила генерации SQL запросов для агента', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Vector Search Instructions - инструкции по векторному поиску
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('VECTOR_SEARCH_INSTRUCTIONS', 'instruction',
    'ИНСТРУКЦИИ ПО ВЕКТОРНОМУ ПОИСКУ:

1. КОГДА ИСПОЛЬЗОВАТЬ:
   - Для текстовых запросов по названию товара (например: "куриная грудка", "говядина")
   - Для поиска по типу товара (например: "полуфабрикаты", "охлажденное мясо")
   - Для семантического поиска (поиск синонимов и связанных понятий)
   - Когда пользователь ищет товары по описанию или характеристикам

2. КОГДА НЕ ИСПОЛЬЗОВАТЬ:
   - Для числовых запросов (цена, вес) - используй SQL запросы
   - Для точных фильтров по поставщику или региону - используй SQL запросы
   - Для комбинаций числовых и текстовых условий - используй SQL запросы

3. ПАРАМЕТРЫ:
   - query: текстовый запрос пользователя (обязательный)
   - k: количество результатов для возврата (по умолчанию 10, максимум 50)
   - require_photo: если True, возвращает только товары с фотографиями

4. РЕЗУЛЬТАТЫ:
   - Результаты отсортированы по релевантности (близости к запросу)
   - Используй параметр k для ограничения количества результатов
   - Если результатов много, выбирай самые релевантные (первые в списке)',
    'Инструкции по использованию векторного поиска', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Photo Sending Instructions - инструкции по отправке фото
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('PHOTO_SENDING_INSTRUCTIONS', 'instruction',
    'ИНСТРУКЦИИ ПО ОТПРАВКЕ ФОТОГРАФИЙ ТОВАРОВ:

1. КОГДА ОТПРАВЛЯТЬ ФОТО:
   - Когда пользователь просит показать товары
   - Когда пользователь спрашивает "покажи фото" или "есть фото?"
   - После успешного поиска товаров, если они есть в результатах
   - В начале разговора (init conversation) - отправляй до 2 фото
   - В обычном разговоре - отправляй до 3 фото

2. КАК ОТПРАВЛЯТЬ:
   - Используй инструмент show_product_photos БЕЗ параметров после получения результатов поиска
   - ID товаров автоматически сохраняются в контекст после вызова vector_search, execute_sql_query, get_random_products
   - НЕ передавай параметр product_ids - просто вызови show_product_photos()
   - Ограничивай количество фото: 2 для init, 3 для обычных запросов

3. ОБРАБОТКА ОШИБОК:
   - ВСЕГДА проверяй результат show_product_photos
   - Если фото НЕ отправились (статус "НЕ ОТПРАВЛЕНО"):
     * НЕ говори что фото отправлены
     * ВСЕГДА предложи товары текстом с информацией о них
     * Сообщи пользователю что фото временно недоступны, но товары есть
   - Если товары не найдены (статус "НЕ НАЙДЕНО"):
     * Сообщи что товары не найдены
     * Предложи альтернативные варианты поиска

4. ВАЖНО:
   - НИКОГДА не говори что фото отправлены, если инструмент вернул "НЕ ОТПРАВЛЕНО"
   - ВСЕГДА предлагай товары текстом, даже если фото не отправились
   - Если фото отправились успешно, можешь упомянуть это в ответе
   - Комбинируй отправку фото с текстовым описанием товаров для лучшего UX',
    'Инструкции по отправке фотографий товаров', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

-- Tool Usage Guidelines - общие инструкции по использованию инструментов
INSERT INTO myaso.rules (rule_name, rule_type, rule_value, description, category)
VALUES 
    ('TOOL_USAGE_GUIDELINES', 'instruction',
    'ОБЩИЕ ИНСТРУКЦИИ ПО ИСПОЛЬЗОВАНИЮ ИНСТРУМЕНТОВ:

1. ПОРЯДОК ИСПОЛЬЗОВАНИЯ:
   - Сначала попробуй понять запрос пользователя
   - Выбери подходящий инструмент:
     * vector_search - для текстового поиска по названию/типу
     * generate_sql_from_text + execute_sql_query - для числовых фильтров (цена, вес, скидка)
     * get_random_products - только как fallback когда ничего не найдено
   - После получения результатов, если есть product_ids, используй show_product_photos для отправки фото

2. КОМБИНАЦИЯ ИНСТРУМЕНТОВ:
   - Можно использовать несколько инструментов последовательно
   - Если vector_search не дал результатов, попробуй SQL запрос
   - Если SQL запрос не дал результатов, используй get_random_products как fallback

3. ИСПОЛЬЗОВАНИЕ show_product_photos:
   - КРИТИЧЕСКИ ВАЖНО: ID товаров АВТОМАТИЧЕСКИ сохраняются в контекст агента после вызова инструментов поиска
   - НЕ извлекай ID из текста ответов инструментов - они уже сохранены автоматически
   - НЕ передавай параметр product_ids в show_product_photos - вызывай БЕЗ параметров: show_product_photos()
   - НЕ придумывай ID самостоятельно (не используй [1, 2, 3] или другие числа)
   - Просто вызови show_product_photos() после получения результатов поиска - ID будут взяты из контекста автоматически

4. ОБРАБОТКА ОШИБОК:
   - Если инструмент вернул ошибку, попробуй другой подход
   - Если все инструменты не дали результатов, используй get_random_products
   - ВСЕГДА предоставляй пользователю полезную информацию, даже если поиск не дал результатов

5. ФОРМАТИРОВАНИЕ ОТВЕТОВ:
   - Представляй товары в читаемом формате
   - Включай: название, поставщик, цену, регион
   - Если есть product_ids, предлагай показать фото
   - Будь дружелюбным и полезным',
    'Общие инструкции по использованию инструментов агента', 'agent')
ON CONFLICT (rule_name) DO UPDATE SET 
    rule_value = EXCLUDED.rule_value,
    description = EXCLUDED.description,
    updated_at = CURRENT_TIMESTAMP;

