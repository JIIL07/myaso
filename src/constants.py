# === Agent ===
MAX_AGENT_ITERATIONS = 100
MAX_AGENT_EXECUTION_TIME = 1800
AGENT_RECURSION_LIMIT = MAX_AGENT_ITERATIONS + 5
DEFAULT_TEMPERATURE = 0.5
TEXT_TO_SQL_TEMPERATURE = 0.1
MAX_LOGS = 500

# === SQL ===
DEFAULT_SQL_LIMIT = 50
MAX_SQL_LIMIT = 100
MAX_SQL_RETRY_ATTEMPTS = 3

# === Vector Search ===
DEFAULT_VECTOR_SEARCH_K = 10
VECTOR_SEARCH_PHOTO_LIMIT = 250
MAX_VECTOR_SEARCH_RESULTS = 50
PHOTO_SEARCH_LIMIT_MULTIPLIER = 5

# === Database ===
TABLE_CONVERSATION_HISTORY = "conversation_history"
TABLE_AGENT_CONTEXT = "agent_context"
TABLE_SYSTEM = "system"
COLUMN_CLIENT_PHONE = "client_phone"
COLUMN_ROLE = "role"
COLUMN_MESSAGE = "message"
COLUMN_CONTEXT_DATA = "context_data"
COLUMN_TOPIC = "topic"
COLUMN_PROMPT = "prompt"
COLUMN_VALUE = "value"
COLUMN_CREATED_AT = "created_at"
DB_POOL_MIN_SIZE = 5
DB_POOL_MAX_SIZE = 20
DB_COMMAND_TIMEOUT = 30.0
DEFAULT_DB_TIMEOUT = 10.0

# === Queue (PGMQ) ===
QUEUE_NAME = "delayed_messages"
DELAY_SECONDS = 900  # 15 минут
QUEUE_CHECK_INTERVAL = 30
VISIBILITY_TIMEOUT = 60
BATCH_SIZE = 10
GRACEFUL_SHUTDOWN_TIMEOUT = 30.0

# === Memory ===
MAX_HISTORY_MESSAGES = 10

# === Prompts (Langfuse names) ===
PROMPT_NAME_SYSTEM_PROMPT = "Системный промт"
PROMPT_NAME_PROFILE = "Профиль"
PROMPT_NAME_PRODUCTS = "Товары"
PROMPT_NAME_OFFER = "Предложение"
PROMPT_NAME_REFLECTOR = "Рефлектор"
PROMPT_NAME_COORDINATOR = "Координатор"
PROMPT_NAME_ERROR_HANDLER = "Обработчик ошибок"
PROMPT_NAME_HUMAN_IN_THE_LOOP = "Позвать человека (HITL)"
PROMPT_NAME_STYLE_EDUARD = "Стиль Эдуард"
PROMPT_NAME_STYLE_POLINA = "Стиль Полина"
PROMPT_NAME_STYLE_MASHA = "Стиль Маша"
PROMPT_NAME_FUNCTION = "function"
PROMPT_NAME_INFO = "info"
PROMPT_NAME_UNCLEAR = "unclear"
PROMPT_CACHE_TTL = 600

# === System Values ===
SYSTEM_VALUE_PRICELIST = "Прайс-лист"
DEFAULT_FIELD_VALUE = "по запросу"

# === Error Messages ===
ERROR_MESSAGE_AGENT_FAILED = "Ой, что-то пошло не так 😔. Попробуйте написать еще раз, пожалуйста!"
ERROR_MESSAGE_PRODUCTS_NOT_FOUND = "Товары по вашему запросу не найдены."
ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND = "Товары с фотографиями по вашему запросу не найдены."
ERROR_MESSAGE_DATABASE_NOT_CONFIGURED = "Не настроено подключение к базе данных."
ERROR_MESSAGE_WHATSAPP_FAILED = "Что-то вотсап барахлит 😔. Напишите позже, пожалуйста!"
ERROR_MESSAGE_TELEGRAM_FAILED = "Что-то телеграм барахлит 😔. Напишите позже, пожалуйста!"

# === HTTP ===
HTTP_TIMEOUT_SECONDS = 10.0
EXCLUDED_PATHS = ["/health", "/docs", "/openapi.json", "/redoc"]
