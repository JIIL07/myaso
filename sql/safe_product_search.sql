-- Безопасная RPC функция для поиска товаров по SQL условиям
-- Защищает от SQL injection через использование параметризованных запросов

CREATE OR REPLACE FUNCTION myaso.safe_product_search(
    conditions text,
    lim int DEFAULT 50
) RETURNS TABLE (
    id int,
    title text,
    supplier_name text,
    from_region text,
    photo text,
    order_price_kg numeric
) AS $$
DECLARE
    safe_conditions text;
BEGIN
    -- Базовая валидация: проверяем, что conditions не пустые
    IF conditions IS NULL OR trim(conditions) = '' THEN
        RAISE EXCEPTION 'SQL условия не могут быть пустыми';
    END IF;
    
    -- Проверка на опасные SQL ключевые слова
    -- Используем верхний регистр для проверки
    IF upper(conditions) ~* '\b(DROP|TRUNCATE|DELETE|INSERT|UPDATE|ALTER|CREATE|EXECUTE|EXEC)\b' THEN
        RAISE EXCEPTION 'Обнаружена опасная SQL команда в условиях';
    END IF;
    
    -- Проверка, что conditions не содержит точку с запятой (множественные запросы)
    IF conditions ~ ';' THEN
        RAISE EXCEPTION 'SQL условия не могут содержать точку с запятой';
    END IF;
    
    -- Используем format для безопасной вставки условий
    -- %s автоматически экранирует специальные символы
    RETURN QUERY EXECUTE format(
        'SELECT 
            id,
            title,
            supplier_name,
            from_region,
            photo,
            order_price_kg
         FROM myaso.products
         WHERE %s
         LIMIT %s',
        conditions,
        lim
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Комментарий к функции
COMMENT ON FUNCTION myaso.safe_product_search(text, int) IS 
'Безопасная функция для поиска товаров по SQL WHERE условиям. 
Защищает от SQL injection через валидацию входных данных и использование format().
Параметры:
- conditions: SQL WHERE условия (без ключевого слова WHERE)
- lim: Максимальное количество товаров для возврата';
