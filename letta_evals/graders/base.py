import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from letta_evals.extractors import extractor_requires_agent_state, get_extractor
from letta_evals.models import AgentState, GradeResult, LettaMessageUnion, Sample


class Grader(ABC):
    """Base interface for graders."""

    def _init_extractor(
        self,
        extractor: str,
        extractor_config: Optional[dict] = None,
        base_dir: Optional[Path] = None,
    ) -> None:
        """Initialize extractor and agent_state requirement. Call from subclass __init__."""
        self.extractor = get_extractor(extractor, extractor_config, base_dir=base_dir)
        self._requires_agent_state = extractor_requires_agent_state(extractor, base_dir=base_dir)

    @property
    def requires_agent_state(self) -> bool:
        """Whether this grader's extractor requires agent_state."""
        return self._requires_agent_state

    def extract(
        self,
        trajectory: List[List[LettaMessageUnion]],
        agent_state: Optional[AgentState] = None,
    ) -> Tuple[Optional[str], float, Optional[Tuple[GradeResult, str]]]:
        """Validate trajectory, run extractor with timing, validate submission.

        Returns:
            (submission, extraction_time, early_return)
            - If early_return is not None, the grader should return it immediately.
            - Otherwise, use submission and extraction_time for scoring.
        """
        if not trajectory or not any(turn for turn in trajectory if turn):
            return None, 0.0, (GradeResult(score=0.0, rationale="Empty trajectory - agent produced no messages"), "")

        t_extract = time.perf_counter()
        submission = self.extractor(trajectory, agent_state=agent_state)
        extraction_time = time.perf_counter() - t_extract

        if not submission:
            return (
                None,
                extraction_time,
                (GradeResult(score=0.0, rationale="Empty submission - extractor found no content"), ""),
            )

        return submission, extraction_time, None

    @abstractmethod
    async def grade(
        self, sample: Sample, trajectory: List[List[LettaMessageUnion]], agent_state: Optional[AgentState] = None
    ) -> Tuple[GradeResult, str]:
        """Grade a trajectory and return the result and extracted submission."""
        pass
