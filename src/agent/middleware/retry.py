from __future__ import annotations

import asyncio
import logging
import random
from typing import Awaitable, Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call

logger = logging.getLogger(__name__)


def create_model_retry_middleware(
    max_retries: int = 2,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retry_on: tuple[type[Exception], ...] | Callable[[Exception], bool] | None = None,
    on_failure: str | Callable[[Exception], ModelResponse] = "error",
) -> Callable:
    @wrap_model_call
    async def model_retry_middleware(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | Awaitable[ModelResponse]],
    ) -> ModelResponse:
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = handler(request)
                if isinstance(result, Awaitable):
                    result = await result
                return result
            except Exception as e:
                last_exception = e

                should_retry = True
                if retry_on is not None:
                    if callable(retry_on):
                        should_retry = retry_on(e)
                    else:
                        should_retry = isinstance(e, retry_on)

                if not should_retry:
                    logger.debug("[ModelRetry] %s not in retry list", type(e).__name__)
                    raise

                if attempt >= max_retries:
                    logger.warning(
                        "[ModelRetry] All attempts exhausted (%d): %s",
                        max_retries + 1, e,
                    )

                    if on_failure == "error":
                        raise
                    elif on_failure == "continue":
                        from langchain_core.messages import AIMessage

                        return ModelResponse(
                            result=[
                                AIMessage(
                                    content="Model call error after %d attempts: %s"
                                    % (max_retries + 1, e)
                                )
                            ]
                        )
                    elif callable(on_failure):
                        return on_failure(e)
                    else:
                        raise

                delay = initial_delay * (backoff_factor ** attempt)
                delay = min(delay, max_delay)

                if jitter:
                    jitter_amount = delay * 0.25
                    delay = delay + random.uniform(-jitter_amount, jitter_amount)
                    delay = max(0, delay)

                logger.info(
                    "[ModelRetry] Attempt %d/%d failed: %s. Retrying in %.2fs",
                    attempt + 1, max_retries + 1, e, delay,
                )

                await asyncio.sleep(delay)

        if last_exception:
            raise last_exception
        raise RuntimeError("[ModelRetry] Unexpected error")

    return model_retry_middleware

