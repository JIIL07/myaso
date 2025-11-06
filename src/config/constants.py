"""Константы для приложения."""

MAX_HISTORY_MESSAGES = 10

VECTOR_SEARCH_LIMIT = 15
DEFAULT_VECTOR_SEARCH_K = 10

DEFAULT_SQL_LIMIT = 15
MAX_SQL_LIMIT = 100

DEFAULT_TEMPERATURE = 0.8
TEXT_TO_SQL_TEMPERATURE = 0.1

MAX_AGENT_ITERATIONS = 5
MAX_AGENT_EXECUTION_TIME = 30

MAX_SQL_RETRY_ATTEMPTS = 3

EMBEDDING_DELAY_SECONDS = 0.1
EMBEDDING_BATCH_SIZE = 10

HTTP_TIMEOUT_SECONDS = 10.0
DB_CONNECTION_TIMEOUT = 10.0
DB_COMMAND_TIMEOUT = 30.0

DANGEROUS_SQL_KEYWORDS = [
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "EXEC",
    "EXECUTE",
]

FORBIDDEN_SQL_PATTERNS = [
    "information_schema",
    "pg_catalog",
    "pg_",
    "myaso.products",
    "::text",
    "::int",
    "::varchar",
    "CAST(",
    "CONVERT(",
]

ALLOWED_SQL_COLUMNS = {
    "id",
    "title",
    "supplier_name",
    "from_region",
    "photo",
    "pricelist_date",
    "package_weight",
    "order_price_kg",
    "min_order_weight_kg",
    "discount",
    "ready_made",
    "package_type",
    "cooled_or_frozen",
    "product_in_package",
}

ALLOWED_SQL_OPERATORS = {
    "=", "!=", "<>", "<", ">", "<=", ">=",
    "AND", "OR", "NOT",
    "IS NULL", "IS NOT NULL",
    "ILIKE", "LIKE",
    "IN", "NOT IN",
    "BETWEEN", "NOT BETWEEN",
}

ALLOWED_SQL_FUNCTIONS = {
    "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
    "NOW()",
    "LOWER",
    "UPPER",
    "TRIM",
}
