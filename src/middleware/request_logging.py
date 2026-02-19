import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.constants import EXCLUDED_PATHS

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in EXCLUDED_PATHS:
            return await call_next(request)

        start_time = time.time()
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        logger.info("[HTTP] %s %s started [%s]", request.method, request.url.path, correlation_id)

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            logger.info(
                "[HTTP] %s %s %s %.3fs [%s]",
                request.method, request.url.path,
                response.status_code, process_time, correlation_id,
            )
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "[HTTP] %s %s error: %s %.3fs [%s]",
                request.method, request.url.path,
                e, process_time, correlation_id,
                exc_info=True,
            )
            raise
