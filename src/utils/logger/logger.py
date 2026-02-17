import json
import logging
import os
from datetime import datetime
from typing import Any

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
            # Используем add_fields для создания log_record
            log_record = {}
            self.add_fields(log_record, record, {})
            log_record["message"] = record.getMessage()
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
        if record.levelno >= logging.ERROR:
            return True
        
        if record.levelno >= logging.WARNING:
            return True
        
        if record.levelno == logging.INFO:
            message = record.getMessage()
            return any(message.startswith(prefix) for prefix in self.IMPORTANT_INFO_PREFIXES)
        
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
    
    # Очищаем все handlers у root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Очищаем handlers у всех существующих логгеров, включая uvicorn
    logger_names = list(logging.Logger.manager.loggerDict.keys())
    
    for logger_name in logger_names:
        logger_obj = logging.getLogger(logger_name)
        if hasattr(logger_obj, 'handlers'):
            for handler in logger_obj.handlers[:]:
                logger_obj.removeHandler(handler)
                handler.close()
        # Не устанавливаем propagate = True для всех, только для root
        if logger_name != "":
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
    
    important_filter = ImportantOnlyFilter()
    console_handler.addFilter(important_filter)

    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    
    # Отключаем propagate для root logger, чтобы избежать дублирования
    root_logger.propagate = False

    logging.getLogger("agents.tools").setLevel(logging.WARNING)
    logging.getLogger("utils.callbacks.langfuse_callback").setLevel(logging.WARNING)
    logging.getLogger("utils.callbacks.reasoning_logger").setLevel(logging.WARNING)  # Убираем избыточные логи
    logging.getLogger("agents.product_agent").setLevel(logging.INFO)

    # Отключаем логирование uvicorn и его компонентов, чтобы избежать дублирования
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.asgi").setLevel(logging.WARNING)
    
    # Отключаем логирование других библиотек
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    _logging_setup_done = True


def log_agent_action(
    action: str,
    client_phone: str | None = None,
    **kwargs: Any,
) -> None:
    """Логирует действие агента.

    Args:
        action: Описание действия
        client_phone: Номер телефона клиента (опционально)
        **kwargs: Дополнительные поля для логирования
    """
    logger = logging.getLogger("agents.product_agent")
    extra = {"action": action, **kwargs}
    if client_phone:
        extra["client_phone"] = client_phone
    logger.info(f"[Agent] {action}", extra=extra)


def log_tool_call(
    tool_name: str,
    client_phone: str | None = None,
    **kwargs: Any,
) -> None:
    """Логирует вызов инструмента.

    Args:
        tool_name: Название инструмента
        client_phone: Номер телефона клиента (опционально)
        **kwargs: Дополнительные поля для логирования
    """
    logger = logging.getLogger("agents.tools")
    extra = {"tool_name": tool_name, **kwargs}
    if client_phone:
        extra["client_phone"] = client_phone
    logger.info(f"[Tool] {tool_name}", extra=extra)


def log_database_operation(
    operation: str,
    table: str | None = None,
    client_phone: str | None = None,
    **kwargs: Any,
) -> None:
    """Логирует операцию с базой данных.

    Args:
        operation: Описание операции
        table: Название таблицы (опционально)
        client_phone: Номер телефона клиента (опционально)
        **kwargs: Дополнительные поля для логирования
    """
    logger = logging.getLogger("database")
    extra = {"operation": operation, **kwargs}
    if table:
        extra["table"] = table
    if client_phone:
        extra["client_phone"] = client_phone
    logger.info(f"[Database] {operation}", extra=extra)
