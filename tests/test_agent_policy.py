"""Tests for agent policy defaults and clamping behavior."""
from __future__ import annotations

from src.agent.product_agent.policy import get_agent_policy


def test_policy_clamps_iterations() -> None:
    policy = get_agent_policy()
    assert policy.clamp_iterations(0) == 1
    assert policy.clamp_iterations(policy.max_iterations_clamp + 1) == policy.max_iterations_clamp


def test_policy_clamps_sql_limit() -> None:
    policy = get_agent_policy()
    assert policy.clamp_sql_limit(0) == 1
    assert policy.clamp_sql_limit(policy.max_sql_limit + 100) == policy.max_sql_limit


def test_photo_search_limit_uses_multiplier_and_cap() -> None:
    policy = get_agent_policy()
    assert policy.photo_search_limit(base_limit=10, require_photo=False) == 10
    expected = min(10 * policy.photo_limit_multiplier, policy.vector_photo_limit)
    assert policy.photo_search_limit(base_limit=10, require_photo=True) == expected
