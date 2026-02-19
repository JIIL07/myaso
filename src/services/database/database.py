import asyncio
import logging
import os
from typing import Optional

import asyncpg

from src.constants import DB_POOL_MIN_SIZE, DB_POOL_MAX_SIZE, DB_COMMAND_TIMEOUT

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None
_lock = asyncio.Lock()


async def get_pool() -> asyncpg.Pool:
    global _pool

    if _pool is not None:
        return _pool

    async with _lock:
        # Double-check после получения блокировки
        if _pool is None:
            db_dsn = os.getenv("POSTGRES_DSN")
            if not db_dsn:
                raise RuntimeError(
                    "POSTGRES_DSN is not set. Provide POSTGRES_DSN in .env"
                )

            try:
                _pool = await asyncpg.create_pool(
                    dsn=db_dsn,
                    min_size=DB_POOL_MIN_SIZE,
                    max_size=DB_POOL_MAX_SIZE,
                    command_timeout=DB_COMMAND_TIMEOUT,
                )
            except Exception as e:
                logger.error("[DB] Ошибка создания pool: %s", e, exc_info=True)
                raise RuntimeError(f"Не удалось создать connection pool: {e}") from e

    return _pool


async def close_pool() -> None:
    global _pool

    if _pool is not None:
        try:
            await _pool.close()
        except Exception as e:
            logger.error("[DB] Ошибка закрытия pool: %s", e, exc_info=True)
        finally:
            _pool = None
