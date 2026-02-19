from dataclasses import dataclass

from langchain.agents import AgentState
from typing_extensions import NotRequired


@dataclass
class ProductAgentContext:
    client_phone: str


class ProductAgentState(AgentState):
    product_ids: NotRequired[list[int]]
    require_photo: NotRequired[bool]
