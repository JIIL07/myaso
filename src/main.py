from fastapi import FastAPI

from src.middleware.cors_middleware import setup_cors
from src.routers import ai_router, health
from src.utils.logger import setup_logging

setup_logging()

app = FastAPI()

setup_cors(app)

app.include_router(ai_router.router)
app.include_router(health.router)
