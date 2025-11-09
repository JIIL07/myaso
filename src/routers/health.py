"""Health check endpoints."""

import logging

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.utils import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter()


async def check_database() -> str:
    """Проверяет подключение к базе данных Supabase.

    Returns:
        "ok" если подключение успешно, иначе "error"
    """
    try:
        supabase = await get_supabase_client()
        result = await supabase.table("products").select("id").limit(1).execute()
        return "ok"
    except Exception as e:
        logging.warning(f"Database health check failed: {e}")
        return "error"


async def check_whatsapp_api() -> str:
    """Проверяет доступность WhatsApp API.

    Returns:
        "ok" если API доступен, иначе "error"
    """
    try:
        if not settings.whatsapp.whatsapp_api_base_url:
            return "not_configured"

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.head(settings.whatsapp.whatsapp_api_base_url)
            if response.status_code < 500:
                return "ok"
            else:
                return "error"
    except Exception as e:
        logging.warning(f"WhatsApp API health check failed: {e}")
        return "error"


@router.get("/health")
async def health_check():
    """Проверяет состояние всех компонентов системы.

    Returns:
        JSON с результатами проверки каждого компонента
    """
    checks = {
        "status": "healthy",
        "database": await check_database(),
        "whatsapp_api": await check_whatsapp_api(),
    }

    component_checks = {k: v for k, v in checks.items() if k != "status"}
    status_code = 200 if all(v == "ok" for v in component_checks.values()) else 503

    return JSONResponse(content=checks, status_code=status_code)

