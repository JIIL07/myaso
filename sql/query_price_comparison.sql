-- Запрос для поиска товаров от поставщика ООО КИТ,
-- у которых цена в таблице products (order_price_kg) выше,
-- чем цена в таблице price_history (price),
-- с учетом сравнения дат

WITH previous_prices AS (
    SELECT 
        ph.product,
        ph.suplier_name,  -- ВАЖНО: в price_history колонка называется suplier_name (с одной 'p')
        ph.date,
        ph.price,
        ROW_NUMBER() OVER (
            PARTITION BY ph.product, ph.suplier_name 
            ORDER BY ph.date DESC
        ) AS rn
    FROM myaso.price_history ph
    WHERE ph.suplier_name = 'ООО КИТ'
      AND ph.date IS NOT NULL
)
SELECT 
    p.id,
    p.title,
    p.supplier_name,
    p.order_price_kg AS current_price,
    pp.price AS previous_price,
    p.pricelist_date AS current_price_date,
    pp.date AS previous_price_date,
    (p.order_price_kg - pp.price) AS price_difference
FROM myaso.products p
JOIN previous_prices pp 
    ON p.title = pp.product 
    AND p.supplier_name = pp.suplier_name
    AND pp.rn = 1  -- последняя цена ДО pricelist_date
WHERE 
    p.supplier_name = 'ООО КИТ'
    AND p.pricelist_date IS NOT NULL
    AND pp.date IS NOT NULL
    AND p.pricelist_date > pp.date  -- убеждаемся, что история — раньше
    AND p.order_price_kg > pp.price  -- основное условие: текущая цена выше предыдущей
ORDER BY price_difference DESC
LIMIT 50;
