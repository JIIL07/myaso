"""Middleware для генерации и передачи correlation ID."""

import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware для добавления correlation ID к каждому запросу."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Обрабатывает запрос, добавляя correlation ID.

        Args:
            request: Входящий HTTP запрос
            call_next: Следующий middleware или handler

        Returns:
            HTTP ответ с correlation ID в заголовках
        """
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        request.state.correlation_id = correlation_id

        response = await call_next(request)

        response.headers["X-Correlation-ID"] = correlation_id

        return response
