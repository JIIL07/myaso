class AgentError(Exception):
    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class AgentTimeoutError(AgentError):
    pass


class AgentExecutionError(AgentError):
    pass
