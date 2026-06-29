from typing import Any, List, Optional


class TargetError(Exception):
    """Exception raised by targets that carries agent context.

    Carries the agent id plus usage recovered from the target stream before
    failure. The runner uses the agent id to fetch any partial server-side
    artifacts (trajectory/token data) so artifact fetching stays centralized.

    ``partial_trajectory`` and ``token_data`` are retained for compatibility
    with older/custom targets, but LettaCodeTarget no longer populates them.
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
