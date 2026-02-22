"""Tests for tool response contract and product id extraction."""
from __future__ import annotations

from src.agent.middleware import _extract_product_ids_from_result
from src.tools._contract import attach_product_ids, fail_response, ok_response


def test_ok_and_fail_response_shape() -> None:
    ok_text, ok_artifact = ok_response("ok", artifact={"foo": "bar"})
    assert ok_text == "ok"
    assert ok_artifact["status"] == "ok"
    assert ok_artifact["error_code"] is None
    assert ok_artifact["foo"] == "bar"

    fail_text, fail_artifact = fail_response("fail", error_code="boom")
    assert fail_text == "fail"
    assert fail_artifact["status"] == "error"
    assert fail_artifact["error_code"] == "boom"


def test_attach_product_ids_and_extract_from_tuple() -> None:
    artifact = attach_product_ids({"kind": "search"}, [1, 2, -1, 0, 3])
    assert artifact["product_ids"] == [1, 2, 3]
    assert _extract_product_ids_from_result(("text", artifact)) == [1, 2, 3]
