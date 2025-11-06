"""Работа с базой данных."""

from .database import get_pool, close_pool

__all__ = ["get_pool", "close_pool"]

