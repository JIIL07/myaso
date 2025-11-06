"""Управление connection pool для PostgreSQL.

Предоставляет singleton connection pool для переиспользования соединений
к базе данных вместо создания нового соединения для каждого запроса.
"""

import os
import logging
from typing import Optional
import asyncpg

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Получает или создает connection pool для PostgreSQL.

    Pool создается один раз при первом вызове и переиспользуется
    для всех последующих запросов.

    Returns:
        asyncpg.Pool: Connection pool для работы с БД

    Raises:
        RuntimeError: Если POSTGRES_DSN не настроен или не удалось создать pool
    """
    global _pool

    if _pool is None:
        db_dsn = os.getenv("POSTGRES_DSN")
        if not db_dsn:
            raise RuntimeError(
                "POSTGRES_DSN is not set. Provide POSTGRES_DSN in .env"
            )

        try:
            _pool = await asyncpg.create_pool(
                dsn=db_dsn,
                min_size=5,
                max_size=20,
                command_timeout=30.0,
            )
            logger.info("Connection pool создан успешно")
        except Exception as e:
            logger.error(f"Ошибка при создании connection pool: {e}", exc_info=True)
            raise RuntimeError(f"Не удалось создать connection pool: {e}") from e

    return _pool


async def close_pool() -> None:
    """Закрывает connection pool.

    Должно вызываться при завершении приложения для корректного
    закрытия всех соединений.
    """
    global _pool

    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Connection pool закрыт")

