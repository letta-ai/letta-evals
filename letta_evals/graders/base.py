from abc import ABC, abstractmethod
from typing import List

from letta_client import LettaMessageUnion

from letta_evals.models import GradeResult, Sample


class Grader(ABC):
    """Base interface for graders."""

    @abstractmethod
    async def grade(self, sample: Sample, trajectory: List[List[LettaMessageUnion]]) -> GradeResult:
        """Grade a trajectory and return the result."""
        pass
