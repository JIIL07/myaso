-- Включить расширение pgmq
CREATE EXTENSION IF NOT EXISTS pgmq;

-- Добавить поле send_message в таблицу clients
ALTER TABLE myaso.clients 
ADD COLUMN IF NOT EXISTS send_message BOOLEAN DEFAULT TRUE;

-- Обновить существующие записи: установить send_message = TRUE для всех клиентов
UPDATE myaso.clients 
SET send_message = TRUE 
WHERE send_message IS NULL;

-- Создать очередь delayed_messages
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pgmq.meta WHERE queue_name = 'delayed_messages'
    ) THEN
        PERFORM pgmq.create('delayed_messages');
    END IF;
END $$;

-- Создать функцию для удаления всех сообщений клиента из очереди
CREATE OR REPLACE FUNCTION myaso.delete_client_messages(
    p_client_phone VARCHAR,
    p_queue_name VARCHAR DEFAULT 'delayed_messages'
)
RETURNS INTEGER AS $$
DECLARE
    v_msg_id BIGINT;
    v_deleted_count INTEGER := 0;
BEGIN
    FOR v_msg_id IN 
        SELECT msg_id 
        FROM pgmq.q_delayed_messages 
        WHERE message->>'client_phone' = p_client_phone
    LOOP
        PERFORM pgmq.delete(p_queue_name, v_msg_id);
        v_deleted_count := v_deleted_count + 1;
    END LOOP;
    
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Создать функцию для очистки истории разговора (оставляет только одно первое сообщение)
CREATE OR REPLACE FUNCTION myaso.clear_conversation_history_keep_one(
    p_client_phone VARCHAR
)
RETURNS INTEGER AS $$
DECLARE
    v_deleted_count INTEGER := 0;
    v_total_count INTEGER := 0;
BEGIN
    -- Подсчитываем общее количество сообщений
    SELECT COUNT(*) INTO v_total_count
    FROM myaso.conversation_history
    WHERE client_phone = p_client_phone;
    
    -- Если сообщений больше 1, удаляем все кроме первого
    IF v_total_count > 1 THEN
        -- Удаляем все сообщения кроме одного первого (по времени создания)
        -- Используем подзапрос для исключения первого сообщения
        DELETE FROM myaso.conversation_history
        WHERE client_phone = p_client_phone
          AND ctid NOT IN (
              SELECT ctid
              FROM myaso.conversation_history
              WHERE client_phone = p_client_phone
              ORDER BY created_at ASC
              LIMIT 1
          );
        
        GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    END IF;
    
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Создать триггер для автоматического удаления сообщений и очистки истории при send_message = false
CREATE OR REPLACE FUNCTION myaso.trigger_delete_messages_on_send_message_false()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.send_message = FALSE AND (OLD.send_message IS NULL OR OLD.send_message = TRUE) THEN
        -- Удаляем сообщения из очереди PGMQ
        PERFORM myaso.delete_client_messages(NEW.phone);
        
        -- Очищаем историю разговора, оставляя только одно первое сообщение
        PERFORM myaso.clear_conversation_history_keep_one(NEW.phone);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Создать триггер на таблице clients
DROP TRIGGER IF EXISTS trigger_delete_messages_on_send_message_false ON myaso.clients;

CREATE TRIGGER trigger_delete_messages_on_send_message_false
    AFTER UPDATE OF send_message ON myaso.clients
    FOR EACH ROW
    WHEN (NEW.send_message = FALSE)
    EXECUTE FUNCTION myaso.trigger_delete_messages_on_send_message_false();

