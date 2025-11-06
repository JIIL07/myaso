import logging
import json
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


def setup_logging():
    """Setup logging for Docker container"""

    import os

    log_format = os.getenv("LOG_FORMAT", "json")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    console_handler = logging.StreamHandler()

    if log_format == "json":
        formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    logging.getLogger("agents.tools").setLevel(logging.INFO)
    logging.getLogger("utils.langfuse_handler").setLevel(logging.INFO)
    logging.getLogger("agents.product_agent").setLevel(logging.INFO)
