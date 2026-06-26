from typing import Any, List, Optional


class TargetError(Exception):
    """Exception raised by targets that carries agent context.

    Also carries any partial trajectory, usage, and token data recovered from
    the server (or the agent's own stream) before the failure, so the runner
    can surface whatever the agent produced on the error path instead of
    discarding it.
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        partial_trajectory: Optional[List[List[Any]]] = None,
        agent_usage: Optional[List[dict]] = None,
        token_data: Optional[List[Any]] = None,
    ):
        super().__init__(message)
        self.agent_id = agent_id
        self.partial_trajectory = partial_trajectory or []
        self.agent_usage = agent_usage
        self.token_data = token_data
