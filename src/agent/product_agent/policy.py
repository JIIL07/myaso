from __future__ import annotations

from dataclasses import dataclass

from src import constants


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int
    backoff_factor: float
    initial_delay: float
    max_delay: float
    jitter: bool
    on_failure: str


@dataclass(frozen=True)
class AgentPolicy:
    max_iterations: int
    max_iterations_clamp: int
    recursion_limit: int
    execution_timeout_seconds: int

    default_temperature: float
    text_to_sql_temperature: float

    default_sql_limit: int
    max_sql_limit: int
    default_vector_k: int
    max_vector_results: int
    vector_photo_limit: int
    photo_limit_multiplier: int

    model_retry: RetryPolicy
    tool_retry: RetryPolicy

    @classmethod
    def from_constants(cls) -> "AgentPolicy":
        return cls(
            max_iterations=constants.MAX_AGENT_ITERATIONS,
            max_iterations_clamp=constants.AGENT_MAX_ITERATIONS_CLAMP,
            recursion_limit=constants.AGENT_RECURSION_LIMIT,
            execution_timeout_seconds=constants.MAX_AGENT_EXECUTION_TIME,
            default_temperature=constants.DEFAULT_TEMPERATURE,
            text_to_sql_temperature=constants.TEXT_TO_SQL_TEMPERATURE,
            default_sql_limit=constants.DEFAULT_SQL_LIMIT,
            max_sql_limit=constants.MAX_SQL_LIMIT,
            default_vector_k=constants.DEFAULT_VECTOR_SEARCH_K,
            max_vector_results=constants.MAX_VECTOR_SEARCH_RESULTS,
            vector_photo_limit=constants.VECTOR_SEARCH_PHOTO_LIMIT,
            photo_limit_multiplier=constants.PHOTO_SEARCH_LIMIT_MULTIPLIER,
            model_retry=RetryPolicy(
                max_retries=constants.MODEL_RETRY_MAX_RETRIES,
                backoff_factor=constants.MODEL_RETRY_BACKOFF_FACTOR,
                initial_delay=constants.MODEL_RETRY_INITIAL_DELAY,
                max_delay=constants.MODEL_RETRY_MAX_DELAY,
                jitter=constants.MODEL_RETRY_JITTER,
                on_failure=constants.MODEL_RETRY_ON_FAILURE,
            ),
            tool_retry=RetryPolicy(
                max_retries=constants.TOOL_RETRY_MAX_RETRIES,
                backoff_factor=constants.TOOL_RETRY_BACKOFF_FACTOR,
                initial_delay=constants.TOOL_RETRY_INITIAL_DELAY,
                max_delay=constants.TOOL_RETRY_MAX_DELAY,
                jitter=constants.TOOL_RETRY_JITTER,
                on_failure=constants.TOOL_ERROR_ON_FAILURE,
            ),
        )

    def clamp_iterations(self, value: int | None = None) -> int:
        iterations = self.max_iterations if value is None else value
        return max(1, min(self.max_iterations_clamp, int(iterations)))

    def clamp_sql_limit(self, value: int | None = None) -> int:
        limit = self.default_sql_limit if value is None else value
        return max(1, min(self.max_sql_limit, int(limit)))

    def clamp_vector_k(self, value: int | None = None) -> int:
        k = self.default_vector_k if value is None else value
        return max(1, min(self.max_vector_results, int(k)))

    def photo_search_limit(self, *, base_limit: int, require_photo: bool) -> int:
        if not require_photo:
            return base_limit
        multiplied = base_limit * self.photo_limit_multiplier
        return min(multiplied, self.vector_photo_limit)


def get_agent_policy() -> AgentPolicy:
    return AgentPolicy.from_constants()
