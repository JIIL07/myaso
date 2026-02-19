import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_not_empty(value: Optional[str], field_name: str, context: str = "") -> bool:
    if not value or not str(value).strip():
        if context:
            logger.warning("[Validator] Empty %s (%s)", field_name, context)
        else:
            logger.warning("[Validator] Empty %s", field_name)
        return False
    return True
