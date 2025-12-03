-- Просмотр всех сообщений в очереди (включая невидимые с delay)
SELECT 
    msg_id,
    read_ct,
    enqueued_at,
    vt as visibility_timeout,
    CASE 
        WHEN vt > NOW() THEN 'Невидимо (delay не истек)'
        ELSE 'Видимо (готово к обработке)'
    END as status,
    message
FROM pgmq.q_delayed_messages
ORDER BY msg_id DESC
LIMIT 50;

-- Просмотр только видимых сообщений (готовых к обработке)
SELECT 
    msg_id,
    read_ct,
    enqueued_at,
    vt as visibility_timeout,
    message
FROM pgmq.q_delayed_messages
WHERE vt <= NOW()
ORDER BY msg_id DESC
LIMIT 50;

-- Просмотр только невидимых сообщений (с активным delay)
SELECT 
    msg_id,
    read_ct,
    enqueued_at,
    vt as visibility_timeout,
    EXTRACT(EPOCH FROM (vt - NOW()))::integer as seconds_until_visible,
    message
FROM pgmq.q_delayed_messages
WHERE vt > NOW()
ORDER BY vt ASC
LIMIT 50;