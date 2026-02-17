"""Типы для ProductAgent - Context и State схемы.

Этот файл содержит определения типов для ProductAgent, которые используются
как в самом агенте, так и в инструментах. Вынесены в отдельный файл для
избежания циклических импортов.
"""

from dataclasses import dataclass
from typing import List

from langchain.agents import AgentState
from typing_extensions import NotRequired


@dataclass
class ProductAgentContext:
    """Контекст для ProductAgent.
    
    Содержит статическую информацию, которая не меняется во время выполнения агента.
    """
    client_phone: str


class ProductAgentState(AgentState):
    """Состояние ProductAgent.
    
    Расширяет базовое AgentState дополнительными полями для отслеживания
    состояния агента во время выполнения.
    """
    product_ids: NotRequired[List[int]]
    require_photo: NotRequired[bool]
