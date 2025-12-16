from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from letta_evals.models import AgentState, GradeResult, LettaMessageUnion, Sample


class Grader(ABC):
    """Base interface for graders."""

    @property
    @abstractmethod
    def requires_agent_state(self) -> bool:
        """Whether this grader requires agent_state for extraction."""
        pass

    @abstractmethod
    async def grade(
        self, sample: Sample, trajectory: List[List[LettaMessageUnion]], agent_state: Optional[AgentState] = None
    ) -> Tuple[GradeResult, str]:
        """Grade a trajectory and return the result and extracted submission."""
        pass
