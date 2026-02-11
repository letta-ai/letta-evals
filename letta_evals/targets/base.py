from abc import ABC, abstractmethod
from typing import Optional

from letta_evals.models import Sample, TargetResult
from letta_evals.visualization.base import ProgressCallback


class TargetError(Exception):
    """Exception raised by targets that carries agent context."""

    def __init__(self, message: str, agent_id: Optional[str] = None):
        super().__init__(message)
        self.agent_id = agent_id


class AbstractAgentTarget(ABC):
    """Base interface for evaluation targets."""

    @abstractmethod
    async def run(self, sample: Sample, progress_callback: Optional[ProgressCallback] = None, **kwargs) -> TargetResult:
        """Run the target on a sample and return result."""
        pass
