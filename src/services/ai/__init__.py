"""AI services — lazy imports to prevent circular dependencies."""

from .customer import CustomerService
from .exceptions import ClientValidationError


def __getattr__(name: str):
    if name == "ConversationService":
        from .conversation import ConversationService

        return ConversationService
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = [
    "ConversationService",
    "CustomerService",
    "ClientValidationError",
]
