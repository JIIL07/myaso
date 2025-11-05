-- Создание схемы (если еще не создана)
CREATE SCHEMA IF NOT EXISTS myaso;

-- Предоставление прав доступа для ролей Supabase
GRANT USAGE ON SCHEMA myaso TO anon, authenticated, service_role;
GRANT ALL ON ALL TABLES IN SCHEMA myaso TO anon, authenticated, service_role;
GRANT ALL ON ALL ROUTINES IN SCHEMA myaso TO anon, authenticated, service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA myaso TO anon, authenticated, service_role;

-- Права по умолчанию для будущих объектов
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA myaso GRANT ALL ON TABLES TO anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA myaso GRANT ALL ON ROUTINES TO anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA myaso GRANT ALL ON SEQUENCES TO anon, authenticated, service_role;
