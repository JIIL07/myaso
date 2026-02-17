"""Health check endpoints."""
import logging

import httpx
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.services.database.database import get_pool

logger = logging.getLogger(__name__)

router = APIRouter()


async def check_database() -> tuple[str, str]:
    """Проверяет подключение к базе данных через connection pool.

    Returns:
        Кортеж (статус, сообщение): ("ok", "") или ("error", "описание ошибки")
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return "ok", ""
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Database health check failed: {error_msg}")
        return "error", error_msg


async def check_queue() -> tuple[str, str]:
    """Проверяет доступность очереди PGMQ.

    Returns:
        Кортеж (статус, сообщение): ("ok", "") или ("error", "описание ошибки")
    """
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT COUNT(*) FROM pgmq.q_delayed_messages LIMIT 1"
            )
            return "ok", ""
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Queue health check failed: {error_msg}")
        return "error", error_msg


async def check_telegram_api() -> tuple[str, str]:
    """Проверяет доступность Telegram API.

    Returns:
        Кортеж (статус, сообщение): ("ok", ""), ("not_configured", "") или ("error", "описание ошибки")
    """
    try:
        if not settings.telegram.telegram_api_base_url:
            return "not_configured", "Telegram API URL not configured"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.head(settings.telegram.telegram_api_base_url)
            if response.status_code < 500:
                return "ok", ""
            else:
                return "error", f"HTTP {response.status_code}"
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Telegram API health check failed: {error_msg}")
        return "error", error_msg


@router.get("/health")
async def health_check():
    """Базовый health check (liveness probe).

    Returns:
        JSON с базовым статусом приложения
    """
    return JSONResponse(
        content={"status": "healthy"},
        status_code=status.HTTP_200_OK,
    )


@router.get("/health/ready")
async def readiness_check():
    """Проверка готовности (readiness probe).

    Проверяет состояние всех зависимостей: БД, очереди, внешние API.

    Returns:
        JSON с результатами проверки каждого компонента
        HTTP 200 если все компоненты здоровы, 503 если есть проблемы
    """
    checks = {}

    db_status, db_message = await check_database()
    checks["database"] = {
        "status": db_status,
        "message": db_message if db_message else None,
    }

    queue_status, queue_message = await check_queue()
    checks["queue"] = {
        "status": queue_status,
        "message": queue_message if queue_message else None,
    }

    telegram_status, telegram_message = await check_telegram_api()
    checks["telegram_api"] = {
        "status": telegram_status,
        "message": telegram_message if telegram_message else None,
    }

    critical_components = ["database", "queue"]
    critical_statuses = [
        checks[component]["status"] for component in critical_components
    ]

    all_healthy = all(status == "ok" for status in critical_statuses)
    overall_status = "ready" if all_healthy else "not_ready"

    status_code = (
        status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        content={
            "status": overall_status,
            "checks": checks,
        },
        status_code=status_code,
    )

