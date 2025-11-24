-- Создание таблицы для хранения контекста агента
CREATE TABLE IF NOT EXISTS myaso.agent_context (
    client_phone TEXT PRIMARY KEY,
    context_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Создание индекса для быстрого поиска по client_phone (уже есть как PRIMARY KEY, но для полноты)
-- Индекс на JSONB для поиска по полям внутри context_data
CREATE INDEX IF NOT EXISTS idx_agent_context_context_data ON myaso.agent_context USING GIN (context_data);

-- Триггер для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION myaso.update_agent_context_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_agent_context_updated_at
    BEFORE UPDATE ON myaso.agent_context
    FOR EACH ROW
    EXECUTE FUNCTION myaso.update_agent_context_updated_at();

-- Предоставление прав доступа
GRANT ALL ON TABLE myaso.agent_context TO anon, authenticated, service_role;

