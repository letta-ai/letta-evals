from typing import Optional


class TargetError(Exception):
    """Exception raised by targets that carries agent context."""

    def __init__(self, message: str, agent_id: Optional[str] = None):
        super().__init__(message)
        self.agent_id = agent_id
