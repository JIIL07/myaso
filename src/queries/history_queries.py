from src.constants import (
    COLUMN_CLIENT_PHONE,
    TABLE_CONVERSATION_HISTORY,
)
from src.services.database.supabase_client import get_supabase_client
from src.services.database.utils import execute_with_timeout


async def get_conversation_history_count(phone: str) -> int:
    try:
        supabase = await get_supabase_client()
        result = await execute_with_timeout(
            supabase.table(TABLE_CONVERSATION_HISTORY)
            .select("id", count="exact")
            .eq(COLUMN_CLIENT_PHONE, phone)
            .execute(),
            operation_name="get_conversation_history_count(%s)" % phone,
        )
        return result.count if result.count else 0
    except Exception as e:
        raise RuntimeError("Error getting history: %s" % e) from e


async def clear_conversation_history(phone: str) -> None:
    try:
        supabase = await get_supabase_client()
        await execute_with_timeout(
            supabase.table(TABLE_CONVERSATION_HISTORY)
            .delete()
            .eq(COLUMN_CLIENT_PHONE, phone)
            .execute(),
            operation_name="clear_conversation_history(%s)" % phone,
        )
    except Exception as e:
        raise RuntimeError("Error clearing history: %s" % e) from e
