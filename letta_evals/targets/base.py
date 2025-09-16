from abc import ABC, abstractmethod
from typing import Optional

from letta_evals.models import Sample, TargetResult
from letta_evals.types import ProgressCallback


class Target(ABC):
    """Base interface for evaluation targets."""

    @abstractmethod
    async def run(self, sample: Sample, progress_callback: Optional[ProgressCallback] = None) -> TargetResult:
        """Run the target on a sample and return result."""
        pass
