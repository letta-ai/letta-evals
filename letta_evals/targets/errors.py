from typing import Any, List, Optional


class TargetError(Exception):
    """Exception raised by targets that carries agent context.

    Carries the agent id plus usage recovered from the target stream before
    failure. The runner uses the agent id to fetch any partial server-side
    trace fields (trajectory/token data) so trace fetching stays centralized.
    On the local backend there is no server, so the target passes the partial
    trajectory it read and the local store it created.
    """

    def __init__(
        self,
        message: str,
        agent_id: Optional[str] = None,
        agent_usage: Optional[List[dict]] = None,
        trajectory: Optional[List[List[Any]]] = None,
        local_backend_dir: Optional[str] = None,
    ):
        super().__init__(message)
        self.agent_id = agent_id
        self.agent_usage = agent_usage
        self.trajectory = trajectory
        self.local_backend_dir = local_backend_dir
