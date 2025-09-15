from abc import ABC, abstractmethod
from typing import List, Tuple

from letta_client import LettaMessageUnion

from letta_evals.models import GradeResult, Sample


class Grader(ABC):
    """Base interface for graders."""

    @abstractmethod
    async def grade(self, sample: Sample, trajectory: List[List[LettaMessageUnion]]) -> Tuple[GradeResult, str]:
        """Grade a trajectory and return the result and extracted submission."""
        raise NotImplementedError
