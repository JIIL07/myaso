"""Управление connection pool для PostgreSQL.

Предоставляет singleton connection pool для переиспользования соединений
к базе данных вместо создания нового соединения для каждого запроса.
"""

import asyncio
import logging
import os
from typing import Optional

import asyncpg

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None
_lock = asyncio.Lock()

DB_POOL_MIN_SIZE = 5
DB_POOL_MAX_SIZE = 20
DB_COMMAND_TIMEOUT = 30.0


async def get_pool() -> asyncpg.Pool:
    """Получает или создает connection pool для PostgreSQL.

    Pool создается один раз при первом вызове и переиспользуется
    для всех последующих запросов. Использует asyncio.Lock для thread-safety
    при параллельных запросах.

    Returns:
        asyncpg.Pool: Connection pool для работы с БД

    Raises:
        RuntimeError: Если POSTGRES_DSN не настроен или не удалось создать pool
    """
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
                logger.error(f"Ошибка при создании connection pool: {e}", exc_info=True)
                raise RuntimeError(f"Не удалось создать connection pool: {e}") from e

    return _pool


async def close_pool() -> None:
    """Закрывает connection pool при shutdown.

    Должно вызываться при завершении приложения для корректного
    закрытия всех соединений с базой данных.
    """
    global _pool

    if _pool is not None:
        try:
            await _pool.close()
        except Exception as e:
            logger.error(f"Ошибка при закрытии connection pool: {e}", exc_info=True)
        finally:
            _pool = None
