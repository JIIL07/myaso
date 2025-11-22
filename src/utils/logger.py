import json
import logging
import os
from datetime import datetime

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Кастомный JSON форматтер для структурированного логирования."""
    
    def __init__(self, *args, **kwargs):
        kwargs.pop("json_ensure_ascii", None)
        super().__init__(*args, **kwargs)

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        log_record["timestamp"] = datetime.utcnow().isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName

        if hasattr(record, "tool_name"):
            log_record["tool_name"] = record.tool_name
        if hasattr(record, "client_phone"):
            log_record["client_phone"] = record.client_phone
        if hasattr(record, "trace_id"):
            log_record["trace_id"] = record.trace_id

    def format(self, record):
        try:
            message = record.getMessage()

            log_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "message": message,
            }

            if hasattr(record, "tool_name"):
                log_record["tool_name"] = record.tool_name
            if hasattr(record, "client_phone"):
                log_record["client_phone"] = record.client_phone
            if hasattr(record, "trace_id"):
                log_record["trace_id"] = record.trace_id

            return json.dumps(log_record, ensure_ascii=False, indent=2)
        except Exception:
            return super().format(record)


class ImportantOnlyFilter(logging.Filter):
    """Фильтр для показа только важных логов.
    
    Показывает:
    - Все ERROR и CRITICAL
    - Все WARNING
    - Только ключевые INFO (с определенными префиксами)
    """
    
    # Префиксы важных INFO сообщений
    IMPORTANT_INFO_PREFIXES = [
        "[ProductAgent.run] ✅",
        "[ProductAgent.run] ⚠️",
        "[processConversation]",
        "[initConversation]",
        "[resetConversation]",
        "ОШИБКА:",
        "ERROR:",
        "CRITICAL:",
    ]
    
    def filter(self, record):
        # Всегда показываем ERROR и CRITICAL
        if record.levelno >= logging.ERROR:
            return True
        
        # Всегда показываем WARNING
        if record.levelno >= logging.WARNING:
            return True
        
        # Для INFO - только важные сообщения
        if record.levelno == logging.INFO:
            message = record.getMessage()
            return any(message.startswith(prefix) for prefix in self.IMPORTANT_INFO_PREFIXES)
        
        # Не показываем DEBUG
        return False


_logging_setup_done = False


def setup_logging():
    """Setup logging for Docker container.

    Вызывается один раз при старте приложения.
    Очищает все существующие handlers перед добавлением новых.
    """
    global _logging_setup_done

    if _logging_setup_done:
        return

    log_format = os.getenv("LOG_FORMAT", "json")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    logger_names = list(logging.Logger.manager.loggerDict.keys())
    
    for logger_name in logger_names:
        logger_obj = logging.getLogger(logger_name)
        if hasattr(logger_obj, 'handlers'):
            logger_obj.handlers.clear()
        if hasattr(logger_obj, 'propagate'):
            logger_obj.propagate = True

    console_handler = logging.StreamHandler()

    if log_format == "json":
        formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    
    # Добавляем фильтр для показа только важных логов
    important_filter = ImportantOnlyFilter()
    console_handler.addFilter(important_filter)

    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)

    # Устанавливаем уровни для специфичных логгеров
    logging.getLogger("agents.tools").setLevel(logging.WARNING)  # Только WARNING и ERROR
    logging.getLogger("utils.callbacks.langfuse_callback").setLevel(logging.WARNING)
    logging.getLogger("utils.callbacks.reasoning_logger").setLevel(logging.WARNING)  # Убираем избыточные логи
    logging.getLogger("agents.product_agent").setLevel(logging.INFO)  # Важные INFO остаются через фильтр

    # Подавляем шумные библиотеки
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _logging_setup_done = True
