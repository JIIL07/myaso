"""Константы для приложения."""

MAX_HISTORY_MESSAGES = 10

VECTOR_SEARCH_LIMIT = 50
DEFAULT_VECTOR_SEARCH_K = 10

DEFAULT_SQL_LIMIT = 50
MAX_SQL_LIMIT = 100

DEFAULT_TEMPERATURE = 0.5
TEXT_TO_SQL_TEMPERATURE = 0.1

# ВАЖНО: Эти значения используются только как fallback, если не заданы в БД через rules
# Рекомендуется настроить их в БД для гибкости без перезапуска приложения
# MAX_AGENT_ITERATIONS = 1000 - очень высокий лимит, может привести к долгим выполнениям
# MAX_AGENT_EXECUTION_TIME = 3600 (1 час) - слишком долго для HTTP запроса
# Рекомендуемые значения для продакшена: 50 итераций, 120 секунд
# Убедитесь, что на уровне nginx/load balancer нет более коротких таймаутов
MAX_AGENT_ITERATIONS = 1000
MAX_AGENT_EXECUTION_TIME = 3600
AGENT_RECURSION_LIMIT = MAX_AGENT_ITERATIONS + 5 

MAX_SQL_RETRY_ATTEMPTS = 3

EMBEDDING_DELAY_SECONDS = 0.1
EMBEDDING_BATCH_SIZE = 10

HTTP_TIMEOUT_SECONDS = 10.0
DB_CONNECTION_TIMEOUT = 10.0
DB_COMMAND_TIMEOUT = 30.0

DANGEROUS_SQL_KEYWORDS = [
    "DROP",
    "TRUNCATE",
    "DELETE",
    "INSERT",
    "EXECUTE",
    "EXEC",
    "UPDATE",
    "ALTER",
    "CREATE",
]

ENABLE_QUERY_REWRITING = False
ENABLE_RERANKING = False
CONTEXT_CACHE_TTL_SECONDS = 300
MAX_HISTORY_FOR_REWRITING = 10
