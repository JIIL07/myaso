"""Константы для AI сервиса."""

PROMPT_TOPIC_WELCOME_MESSAGE = "Вступительное сообщение"
PROMPT_TOPIC_SQL_GENERATION_RULES = "SQL Generation Rules"
PROMPT_TOPIC_VECTOR_SEARCH_INSTRUCTIONS = "Vector Search Instructions"
PROMPT_TOPIC_PHOTO_SENDING_INSTRUCTIONS = "Photo Sending Instructions"
PROMPT_TOPIC_TOOL_USAGE_GUIDELINES = "Tool Usage Guidelines"

SYSTEM_VALUE_PRICELIST = "Прайс-лист"
SYSTEM_VALUE_MARKUP_OVER_100 = "Наценка на кг/руб (>100 руб)"

DEFAULT_FIELD_VALUE = "по запросу"

ERROR_MESSAGE_AGENT_FAILED = "Ой, что-то пошло не так 😔. Попробуйте написать еще раз, пожалуйста!"
ERROR_MESSAGE_PRODUCTS_NOT_FOUND = "Товары по вашему запросу не найдены."
ERROR_MESSAGE_PRODUCTS_WITH_PHOTOS_NOT_FOUND = "Товары с фотографиями по вашему запросу не найдены."
ERROR_MESSAGE_DATABASE_NOT_CONFIGURED = "Не настроено подключение к базе данных."

GREETING_WORDS = [
    "привет",
    "здравствуй",
    "здравствуйте",
    "добрый день",
    "добрый вечер",
    "доброе утро",
    "доброй ночи",
    "доброго дня",
    "доброго вечера",
    "доброго утра",
    "здорово",
    "салют",
    "хай",
    "hi",
    "hello",
    "доброго времени суток",
    "приветствую",
    "добро пожаловать",
]

EMPTY_VALUES = ["не указано", "null", "none", ""]

MAX_AGENT_ITERATIONS = 100
MAX_AGENT_EXECUTION_TIME = 120
AGENT_RECURSION_LIMIT = MAX_AGENT_ITERATIONS + 5

DEFAULT_TEMPERATURE = 0.5
TEXT_TO_SQL_TEMPERATURE = 0.1

DEFAULT_SQL_LIMIT = 50
MAX_SQL_LIMIT = 100
MAX_SQL_RETRY_ATTEMPTS = 3

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
