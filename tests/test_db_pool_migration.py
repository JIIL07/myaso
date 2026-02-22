from __future__ import annotations

from typing import Any

import pytest

from src.queries.clients_queries import get_client_by_phone
from src.queries.history_queries import clear_conversation_history, get_conversation_history_count
from src.services.ai.prompt import get_all_system_values, get_system_value
from src.services.memory.conversation_memory import PostgresConversationMemory


class _AcquireCtx:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    async def __aenter__(self) -> "_FakeConn":
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakePool:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)


class _FakeConn:
    def __init__(self) -> None:
        self.fetchrow_result: Any = None
        self.fetchval_result: Any = None
        self.fetch_result: Any = []
        self.last_query: str = ""
        self.last_args: tuple[Any, ...] = ()
        self.executemany_args: tuple[str, list[tuple[Any, ...]]] | None = None

    async def fetchrow(self, query: str, *args: Any):
        self.last_query = query
        self.last_args = args
        return self.fetchrow_result

    async def fetchval(self, query: str, *args: Any):
        self.last_query = query
        self.last_args = args
        return self.fetchval_result

    async def fetch(self, query: str, *args: Any):
        self.last_query = query
        self.last_args = args
        return self.fetch_result

    async def execute(self, query: str, *args: Any):
        self.last_query = query
        self.last_args = args
        return "OK"

    async def executemany(self, query: str, args: list[tuple[Any, ...]]):
        self.executemany_args = (query, args)
        return None


@pytest.mark.asyncio
async def test_get_client_by_phone_uses_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.queries import clients_queries

    conn = _FakeConn()
    conn.fetchrow_result = {"phone": "+79990001122", "name": "Test"}
    
    async def _fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(clients_queries, "get_pool", _fake_get_pool)

    result = await get_client_by_phone("+79990001122")

    assert result is not None
    assert result["name"] == "Test"
    assert "FROM myaso.clients" in conn.last_query


@pytest.mark.asyncio
async def test_history_queries_use_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.queries import history_queries

    conn = _FakeConn()
    conn.fetchval_result = 3

    async def _fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(history_queries, "get_pool", _fake_get_pool)

    count = await get_conversation_history_count("+79990001122")
    await clear_conversation_history("+79990001122")

    assert count == 3
    assert "DELETE FROM myaso.conversation_history" in conn.last_query


@pytest.mark.asyncio
async def test_prompt_system_values_use_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.ai import prompt as prompt_module

    conn = _FakeConn()
    conn.fetchrow_result = {"value": "https://example.org/price.xlsx"}
    conn.fetch_result = [{"topic": "Прайс-лист", "value": "url"}]

    async def _fake_get_pool():
        return _FakePool(conn)

    monkeypatch.setattr(prompt_module, "get_pool", _fake_get_pool)

    value = await get_system_value("Прайс-лист")
    values = await get_all_system_values()

    assert value == "https://example.org/price.xlsx"
    assert values == {"Прайс-лист": "url"}
    assert "FROM myaso.system" in conn.last_query


@pytest.mark.asyncio
async def test_postgres_memory_create_warms_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.services.memory import conversation_memory as memory_module

    called = {"value": False}

    async def _fake_get_pool():
        called["value"] = True
        return _FakePool(_FakeConn())

    monkeypatch.setattr(memory_module, "get_pool", _fake_get_pool)

    memory = await PostgresConversationMemory.create("+79990001122")

    assert called["value"] is True
    assert memory.async_initialized is True
