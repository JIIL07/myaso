from src.constants import (
    COLUMN_CLIENT_PHONE,
    TABLE_CONVERSATION_HISTORY,
)
from src.services.database.database import get_pool


async def get_conversation_history_count(phone: str) -> int:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM myaso.%s
                WHERE %s = $1
                """
                % (TABLE_CONVERSATION_HISTORY, COLUMN_CLIENT_PHONE),
                phone,
            )
        return int(count or 0)
    except Exception as e:
        raise RuntimeError("Error getting history: %s" % e) from e


async def clear_conversation_history(phone: str) -> None:
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                DELETE FROM myaso.%s
                WHERE %s = $1
                """
                % (TABLE_CONVERSATION_HISTORY, COLUMN_CLIENT_PHONE),
                phone,
            )
    except Exception as e:
        raise RuntimeError("Error clearing history: %s" % e) from e
