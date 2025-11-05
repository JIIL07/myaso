-- 1. Универсальная функция для логирования цены
CREATE OR REPLACE FUNCTION myaso.log_price_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- При вставке всегда записываем цену
        INSERT INTO myaso.price_history (product, date, price, suplier_name)
        VALUES (
            NEW.title,
            NOW(),
            NEW.order_price_kg,
            NEW.supplier_name
        );
    ELSIF TG_OP = 'UPDATE' THEN
        -- При обновлении — только если цена изменилась
        IF OLD.order_price_kg IS DISTINCT FROM NEW.order_price_kg THEN
            INSERT INTO myaso.price_history (product, date, price, suplier_name)
            VALUES (
                NEW.title,
                NOW(),
                NEW.order_price_kg,
                NEW.supplier_name
            );
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. Триггер на INSERT
CREATE TRIGGER trigger_log_price_on_insert
    AFTER INSERT ON myaso.products
    FOR EACH ROW
    EXECUTE FUNCTION myaso.log_price_change();

-- 3. Триггер на UPDATE (только если меняется order_price_kg — опционально, но эффективно)
CREATE TRIGGER trigger_log_price_on_update
    AFTER UPDATE OF order_price_kg ON myaso.products
    FOR EACH ROW
    EXECUTE FUNCTION myaso.log_price_change();