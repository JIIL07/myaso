from typing import Any, Optional

from pydantic import BaseModel


class SuccessResponse(BaseModel):
    success: bool = True


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Any] = None
