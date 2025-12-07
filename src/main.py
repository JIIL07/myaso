import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.middleware.cors_middleware import setup_cors
from src.routers import ai_router, health
from src.utils.logger import setup_logging
from src.workers.queue_worker import start_queue_worker

setup_logging()

logger = logging.getLogger(__name__)

# Глобальная переменная для хранения задачи воркера
queue_worker_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения: запуск и остановка воркера очереди."""
    global queue_worker_task
    
    # Startup: запускаем воркер очереди
    try:
        queue_worker_task = await start_queue_worker()
        logger.info("[main] Воркер очереди запущен при старте приложения")
    except Exception as e:
        logger.error(
            f"[main] Ошибка при запуске воркера очереди: {e}", exc_info=True
        )
    
    yield
    
    # Shutdown: останавливаем воркер очереди
    if queue_worker_task and not queue_worker_task.done():
        logger.info("[main] Останавливаем воркер очереди...")
        queue_worker_task.cancel()
        try:
            await queue_worker_task
        except asyncio.CancelledError:
            pass
        logger.info("[main] Воркер очереди остановлен")
    
    # Закрываем connection pool
    try:
        from src.database import close_pool
        await close_pool()
        logger.info("[main] Connection pool закрыт")
    except Exception as e:
        logger.error(f"[main] Ошибка при закрытии connection pool: {e}", exc_info=True)


app = FastAPI(lifespan=lifespan)

setup_cors(app)

app.include_router(ai_router.router)
app.include_router(health.router)
