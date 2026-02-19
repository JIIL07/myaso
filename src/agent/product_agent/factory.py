from __future__ import annotations

import threading
from typing import Any, Literal, TypeVar, overload

from .base_agent import BaseAgent
from .product_agent import ProductAgent


TAgent = TypeVar("TAgent", bound=BaseAgent)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_value(v) for v in value)
    return value


def _build_cache_key(name: str, config: dict[str, Any]) -> tuple[str, Any]:
    override_key = config.get("cache_key")
    if override_key is not None:
        return name, override_key
    return name, _freeze_value(config)


class AgentFactory:

    _instance: AgentFactory | None = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self.registered_agents: dict[str, type[BaseAgent]] = {}
        self._instances: dict[tuple[str, Any], BaseAgent] = {}
        self.register_agent("product", ProductAgent)

    @classmethod
    def instance(cls) -> AgentFactory:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_agent(self, name: str, agent_class: type[BaseAgent]) -> None:
        self.registered_agents[name] = agent_class

    @overload
    def get_agent(self, name: Literal["product"], config: dict[str, Any], *, use_cache: bool = True) -> ProductAgent: ...

    @overload
    def get_agent(self, name: str, config: dict[str, Any], *, use_cache: bool = True) -> BaseAgent: ...

    def get_agent(self, name: str, config: dict[str, Any], *, use_cache: bool = True) -> BaseAgent:
        if name not in self.registered_agents:
            raise KeyError("Agent '%s' is not registered" % name)

        if not use_cache:
            agent_class = self.registered_agents[name]
            return agent_class(**(config or {}))

        cache_key = _build_cache_key(name, config or {})
        if cache_key in self._instances:
            return self._instances[cache_key]

        with self._lock:
            if cache_key in self._instances:
                return self._instances[cache_key]
            agent_class = self.registered_agents[name]
            instance = agent_class(**(config or {}))
            self._instances[cache_key] = instance
            return instance

    def create_product_agent(self, config: dict[str, Any], *, use_cache: bool = True) -> ProductAgent:
        agent = self.get_agent("product", config, use_cache=use_cache)
        if not isinstance(agent, ProductAgent):
            raise TypeError("Registered 'product' agent is not a ProductAgent")
        return agent
