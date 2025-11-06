from __future__ import annotations

from typing import Any, Dict, Tuple, Type, TypeVar
from threading import RLock

from .base_agent import BaseAgent
from .product_agent import ProductAgent


TAgent = TypeVar("TAgent", bound=BaseAgent)


def _freeze_value(value: Any) -> Any:
    """Возвращает хэшируемое представление произвольного значения.

    Рекурсивно преобразует словари, списки, кортежи и множества в кортежи
    для создания стабильного хэшируемого ключа кэша по конфигурации агента.

    Используется для кэширования агентов с одинаковой конфигурацией.

    Args:
        value: Произвольное значение (dict, list, tuple, set, или примитив)

    Returns:
        Хэшируемое представление значения:
        - dict -> tuple(sorted items)
        - list/tuple/set -> tuple(frozen items)
        - примитивы -> без изменений
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze_value(v) for v in value)
    return value


def _build_cache_key(name: str, config: Dict[str, Any]) -> Tuple[str, Any]:
    """Создаёт ключ кэша для агента.

    Если в конфиге указан явный cache_key, использует его.
    Иначе создаёт ключ из имени агента и замороженного конфига.

    Args:
        name: Имя агента (например, "product")
        config: Словарь конфигурации агента

    Returns:
        Кортеж (name, cache_key) для использования в качестве ключа кэша
    """
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
        self.register_agent("product", ProductAgent)

    @classmethod
    def instance(cls) -> "AgentFactory":
        """Возвращает единственный экземпляр фабрики (singleton)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_agent(self, name: str, agent_class: Type[BaseAgent]) -> None:
        """Регистрирует класс агента под указанным именем.

        Args:
            name: Имя агента для регистрации
            agent_class: Класс агента, наследующийся от BaseAgent
        """
        self.registered_agents[name] = agent_class

    def get_agent(self, name: str, config: Dict[str, Any]) -> BaseAgent:
        """Возвращает (создаёт при необходимости) агента по имени и конфигу.

        Реализует singleton-per-config: один инстанс на уникальный ключ.

        Args:
            name: Имя зарегистрированного агента
            config: Словарь конфигурации агента

        Returns:
            Экземпляр агента

        Raises:
            KeyError: Если агент с таким именем не зарегистрирован
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
        """Создаёт или возвращает `ProductAgent` с учётом единичности по конфигу.

        Args:
            config: Словарь конфигурации для ProductAgent

        Returns:
            Экземпляр ProductAgent

        Raises:
            TypeError: Если зарегистрированный агент не является ProductAgent
        """
        agent = self.get_agent("product", config)
        if not isinstance(agent, ProductAgent):
            raise TypeError("Registered 'product' agent is not a ProductAgent")
        return agent
