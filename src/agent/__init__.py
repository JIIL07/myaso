"""Agent package — lazy imports to prevent circular dependencies."""

from src.services.memory import PostgresConversationMemory


def __getattr__(name: str):
    if name == "BaseAgent":
        from .product_agent.base_agent import BaseAgent

        return BaseAgent
    if name == "ProductAgent":
        from .product_agent.product_agent import ProductAgent

        return ProductAgent
    if name == "AgentFactory":
        from .product_agent.factory import AgentFactory

        return AgentFactory
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = [
    "BaseAgent",
    "ProductAgent",
    "AgentFactory",
    "PostgresConversationMemory",
]
