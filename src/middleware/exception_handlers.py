"""Централизованные обработчики исключений для FastAPI."""

import logging
from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.entities import ErrorResponse

logger = logging.getLogger(__name__)


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Обработчик для HTTP исключений.

    Args:
        request: Запрос FastAPI
        exc: HTTP исключение

    Returns:
        JSON ответ с ошибкой
    """
    logger.warning(
        f"HTTP {exc.status_code} error on {request.method} {request.url.path}: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(success=False, error=str(exc.detail)).model_dump(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Обработчик для ошибок валидации Pydantic.

    Args:
        request: Запрос FastAPI
        exc: Ошибка валидации

    Returns:
        JSON ответ с деталями ошибок валидации
    """
    errors = exc.errors()
    error_messages = [f"{err['loc']}: {err['msg']}" for err in errors]
    error_detail = "; ".join(error_messages)
    
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {error_detail}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            success=False,
            error="Validation error",
            details=errors if len(errors) <= 5 else errors[:5],  # Ограничиваем количество деталей
        ).model_dump(),
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Обработчик для необработанных исключений.

    Args:
        request: Запрос FastAPI
        exc: Исключение

    Returns:
        JSON ответ с общей ошибкой
    """
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            success=False, error="Internal server error"
        ).model_dump(),
    )
