import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from src.routers import ai_router
from src.utils.logger import setup_logging
from src.config.settings import settings
from src.utils.supabase_client import get_supabase_client
from langfuse import Langfuse
import httpx

setup_logging()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ai_router.router)


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


async def check_langfuse() -> str:
    """Проверяет подключение к LangFuse.

    Returns:
        "ok" если подключение успешно, иначе "error"
    """
    try:
        if not settings.langfuse.langfuse_public_key or not settings.langfuse.langfuse_secret_key:
            return "not_configured"

        langfuse = Langfuse(
            public_key=settings.langfuse.langfuse_public_key,
            secret_key=settings.langfuse.langfuse_secret_key,
            host=settings.langfuse.langfuse_host,
        )
        if hasattr(langfuse, "client") and hasattr(langfuse.client, "traces"):
            langfuse.client.traces.list(limit=1)
        return "ok"
    except Exception as e:
        logging.warning(f"LangFuse health check failed: {e}")
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


@app.get("/health")
async def health_check():
    """Проверяет состояние всех компонентов системы.

    Returns:
        JSON с результатами проверки каждого компонента
    """
    checks = {
        "status": "healthy",
        "database": await check_database(),
        "langfuse": await check_langfuse(),
        "whatsapp_api": await check_whatsapp_api(),
    }

    component_checks = {k: v for k, v in checks.items() if k != "status"}
    status_code = 200 if all(v == "ok" for v in component_checks.values()) else 503

    return JSONResponse(content=checks, status_code=status_code)
