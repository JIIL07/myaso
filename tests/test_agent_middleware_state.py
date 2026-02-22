"""Tests for middleware state updates."""
from __future__ import annotations

from src.agent.middleware import _extract_product_ids_from_result


def test_extract_product_ids_from_dict_artifact() -> None:
    result = _extract_product_ids_from_result(("ok", {"product_ids": [10, 20, 30]}))
    assert result == [10, 20, 30]
