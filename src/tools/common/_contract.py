from __future__ import annotations

from typing import Any


def ok_response(
    content: str,
    *,
    artifact: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    body = dict(artifact or {})
    body.setdefault("status", "ok")
    body.setdefault("error_code", None)
    return content, body


def fail_response(
    content: str,
    *,
    error_code: str,
    artifact: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    body = dict(artifact or {})
    body["status"] = "error"
    body["error_code"] = error_code
    return content, body


def attach_product_ids(
    artifact: dict[str, Any] | None,
    product_ids: list[int],
) -> dict[str, Any]:
    body = dict(artifact or {})
    body["product_ids"] = [int(pid) for pid in product_ids if isinstance(pid, int) and pid > 0]
    return body

