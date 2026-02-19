import json
import logging
import os
from datetime import datetime

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
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
        for attr in ("tool_name", "client_phone", "trace_id"):
            if hasattr(record, attr):
                log_record[attr] = getattr(record, attr)

    def format(self, record):
        try:
            log_record = {}
            self.add_fields(log_record, record, {})
            log_record["message"] = record.getMessage()
            return json.dumps(log_record, ensure_ascii=False, indent=2)
        except Exception:
            return super().format(record)


class ImportantOnlyFilter(logging.Filter):
    IMPORTANT_INFO_PREFIXES = [
        "[ProductAgent.run] ✅",
        "[ProductAgent.run] ⚠️",
        "[processConversation]",
        "[initConversation]",
        "[resetConversation]",
    ]

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno == logging.INFO:
            message = record.getMessage()
            return any(message.startswith(p) for p in self.IMPORTANT_INFO_PREFIXES)
        return False


_logging_setup_done = False


def setup_logging():
    global _logging_setup_done
    if _logging_setup_done:
        return

    log_format = os.getenv("LOG_FORMAT", "json")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger_obj = logging.getLogger(logger_name)
        if hasattr(logger_obj, "handlers"):
            for handler in logger_obj.handlers[:]:
                logger_obj.removeHandler(handler)
                handler.close()
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
    console_handler.addFilter(ImportantOnlyFilter())

    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    root_logger.propagate = False

    # Suppress noisy libraries
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "uvicorn.asgi",
                 "urllib3", "httpx", "langchain", "openai",
                 "agents.tools", "utils.callbacks.langfuse_callback",
                 "utils.callbacks.reasoning_logger"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger("agents.product_agent").setLevel(logging.INFO)
    _logging_setup_done = True
