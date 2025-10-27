from abc import ABC, abstractmethod
from typing import Optional

from letta_evals.models import Sample, TargetResult
from letta_evals.visualization.base import ProgressCallback


class AbstractAgentTarget(ABC):
    """Base interface for evaluation targets."""

    @abstractmethod
    async def run(self, sample: Sample, progress_callback: Optional[ProgressCallback] = None, **kwargs) -> TargetResult:
        """Run the target on a sample and return result."""
        pass
