import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.services.callbacks.langfuse_callback import (
    flush_langfuse,
    is_langfuse_enabled,
    create_langfuse_callback_handler,
)
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
    """Application lifespan: start and stop queue worker, flush resources."""
    global queue_worker_task

    # Initialize and verify Langfuse connection
    if is_langfuse_enabled():
        try:
            handler = create_langfuse_callback_handler()
            if handler:
                logger.info("[main] Langfuse CallbackHandler initialized successfully")
            else:
                logger.warning("[main] Langfuse enabled but CallbackHandler creation failed")
        except Exception as e:
            logger.error("[main] Error initializing Langfuse: %s", e, exc_info=True)
    else:
        logger.info("[main] Langfuse is disabled")

    try:
        queue_worker_task = await start_queue_worker()
        logger.info("[main] Queue worker started at app startup")
    except Exception as e:
        logger.error("[main] Error starting queue worker: %s", e, exc_info=True)

    yield

    # Graceful shutdown
    if queue_worker_task and not queue_worker_task.done():
        logger.info("[main] Stopping queue worker...")
        try:
            await graceful_shutdown_worker()
        except Exception as e:
            logger.error("[main] Error during graceful shutdown: %s", e, exc_info=True)
            queue_worker_task.cancel()
            try:
                await queue_worker_task
            except asyncio.CancelledError:
                pass
        logger.info("[main] Queue worker stopped")

    try:
        from src.services.database.database import close_pool
        await close_pool()
        logger.info("[main] Connection pool closed")
    except Exception as e:
        logger.error("[main] Error closing connection pool: %s", e, exc_info=True)

    # Flush Langfuse events
    try:
        flush_langfuse()
        logger.info("[main] Langfuse flushed")
    except Exception as e:
        logger.error("[main] Error flushing Langfuse: %s", e, exc_info=True)


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
