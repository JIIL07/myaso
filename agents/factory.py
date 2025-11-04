from __future__ import annotations

from typing import Any, Dict, Tuple, Type, TypeVar
from threading import RLock

from .base_agent import BaseAgent
from .product_agent import ProductAgent


TAgent = TypeVar("TAgent", bound=BaseAgent)


def _freeze_value(value: Any) -> Any:
    """Возвращает хэшируемое представление произвольного значения.

    Нужен для построения стабильного ключа кэша по `config`.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_value(v) for v in value)
    return value


def _build_cache_key(name: str, config: Dict[str, Any]) -> Tuple[str, Any]:
    override_key = config.get("cache_key")
    if override_key is not None:
        return name, override_key
    return name, _freeze_value(config)


class AgentFactory:
    """Фабрика для создания и переиспользования агентов.

    Возможности:
    - Регистрация новых типов агентов по имени
    - Создание агента по имени и конфигу
    - Переиспользование (singleton per config)

    Как добавить нового агента:
    1) Создайте класс-агент, унаследованный от `BaseAgent`, например `SupportAgent`.
    2) Импортируйте класс в модуль, где будете вызывать фабрику.
    3) Зарегистрируйте класс:
       `AgentFactory.instance().register_agent("support", SupportAgent)`
    4) Получайте агента:
       `agent = AgentFactory.instance().get_agent("support", config={...})`
    """

    _instance: "AgentFactory" | None = None
    _lock = RLock()

    def __init__(self) -> None:
        self.registered_agents: Dict[str, Type[BaseAgent]] = {}
        self._instances: Dict[Tuple[str, Any], BaseAgent] = {}
        # Регистрируем стандартные агенты
        self.register_agent("product", ProductAgent)

    @classmethod
    def instance(cls) -> "AgentFactory":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_agent(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """Регистрирует класс агента под указанным именем."""
        self.registered_agents[name] = agent_class

    def get_agent(self, name: str, config: Dict[str, Any]) -> BaseAgent:
        """Возвращает (создаёт при необходимости) агента по имени и конфигу.

        Реализует singleton-per-config: один инстанс на уникальный ключ.
        """
        if name not in self.registered_agents:
            raise KeyError(f"Agent '{name}' is not registered")

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

    def create_product_agent(self, config: Dict[str, Any]) -> ProductAgent:
        """Создаёт или возвращает `ProductAgent` с учётом единичности по конфигу."""
        agent = self.get_agent("product", config)
        if not isinstance(agent, ProductAgent):
            # Защита от конфликта регистраций под одним именем
            raise TypeError("Registered 'product' agent is not a ProductAgent")
        return agent
