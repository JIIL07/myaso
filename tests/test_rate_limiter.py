"""Tests for src.services.agent_queue.rate_limiter."""
from __future__ import annotations

import pytest

from src.services.agent_queue.rate_limiter import RateLimiter


@pytest.mark.asyncio
class TestRateLimiter:
    async def test_initial_state_is_available(self) -> None:
        rl = RateLimiter(max_concurrent=1)
        assert rl.is_available() is True

    async def test_acquire_makes_unavailable(self) -> None:
        rl = RateLimiter(max_concurrent=1)
        await rl.acquire(task_id="t1")
        assert rl.is_available() is False

    async def test_release_makes_available_again(self) -> None:
        rl = RateLimiter(max_concurrent=1)
        await rl.acquire(task_id="t1")
        await rl.release()
        assert rl.is_available() is True

    async def test_status_available(self) -> None:
        rl = RateLimiter(max_concurrent=2)
        status = rl.get_status()
        assert status["available"] is True
        assert status["available_slots"] == 2
        assert status["current_task"] is None

    async def test_status_after_acquire(self) -> None:
        rl = RateLimiter(max_concurrent=1)
        await rl.acquire(task_id="my_task")
        status = rl.get_status()
        assert status["available"] is False
        assert status["current_task"] == "my_task"

    async def test_multiple_slots(self) -> None:
        rl = RateLimiter(max_concurrent=3)
        await rl.acquire(task_id="t1")
        assert rl.is_available() is True  # 2 slots left
        await rl.acquire(task_id="t2")
        assert rl.is_available() is True  # 1 slot left
        await rl.acquire(task_id="t3")
        assert rl.is_available() is False  # 0 slots left
