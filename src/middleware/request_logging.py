"""Middleware для структурированного логирования HTTP запросов."""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

EXCLUDED_PATHS = ["/health", "/docs", "/openapi.json", "/redoc"]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware для логирования всех HTTP запросов."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Обрабатывает запрос, логируя информацию о нем.

        Args:
            request: Входящий HTTP запрос
            call_next: Следующий middleware или handler

        Returns:
            HTTP ответ
        """
        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)

        start_time = time.time()

        correlation_id = getattr(request.state, "correlation_id", "unknown")

        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"[correlation_id={correlation_id}]"
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Time: {process_time:.3f}s "
                f"[correlation_id={correlation_id}]"
            )

            return response
        except Exception as e:
            process_time = time.time() - start_time

            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"Error: {str(e)} "
                f"Time: {process_time:.3f}s "
                f"[correlation_id={correlation_id}]",
                exc_info=True,
            )
            raise
