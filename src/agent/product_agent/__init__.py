"""LangChain агенты — lazy imports to prevent circular dependencies."""


def __getattr__(name: str):
    if name == "BaseAgent":
        from .base_agent import BaseAgent

        return BaseAgent
    if name == "ProductAgent":
        from .product_agent import ProductAgent

        return ProductAgent
    if name == "AgentFactory":
        from .factory import AgentFactory

        return AgentFactory
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


__all__ = ["BaseAgent", "ProductAgent", "AgentFactory"]
