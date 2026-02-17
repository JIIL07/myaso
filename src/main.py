import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.services.queue.worker import graceful_shutdown_worker, start_queue_worker
from src.middleware.correlation_id import CorrelationIDMiddleware
from src.middleware.cors_middleware import setup_cors
from src.middleware.exception_handlers import (
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from src.middleware.rate_limiter import setup_rate_limiter
from src.middleware.request_logging import RequestLoggingMiddleware
from src.routes import ai_router, health, router
from src.utils.logger.logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

queue_worker_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения: запуск и остановка воркера очереди."""
    global queue_worker_task
    
    try:
        queue_worker_task = await start_queue_worker()
        logger.info("[main] Воркер очереди запущен при старте приложения")
    except Exception as e:
        logger.error(
            f"[main] Ошибка при запуске воркера очереди: {e}", exc_info=True
        )
    
    yield
    
    if queue_worker_task and not queue_worker_task.done():
        logger.info("[main] Останавливаем воркер очереди...")
        try:
            await graceful_shutdown_worker()
        except Exception as e:
            logger.error(f"[main] Ошибка при graceful shutdown воркера: {e}", exc_info=True)
            # Fallback на cancel если graceful shutdown не удался
            queue_worker_task.cancel()
            try:
                await queue_worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[main] Воркер очереди остановлен")
    
    try:
        from src.services.database.database import close_pool
        await close_pool()
        logger.info("[main] Connection pool закрыт")
    except Exception as e:
        logger.error(f"[main] Ошибка при закрытии connection pool: {e}", exc_info=True)


app = FastAPI(lifespan=lifespan)

app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RequestLoggingMiddleware)

setup_cors(app)
setup_rate_limiter(app)

app.include_router(ai_router.router)
app.include_router(health.router)
app.include_router(router.router)