from __future__ import annotations

from typing import Any

import asyncpg


def records_to_json(records: list[asyncpg.Record]) -> list[dict[str, Any]]:
    return [dict(record) for record in records]
