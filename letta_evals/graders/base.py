import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from letta_evals.models import GradeResult, LettaMessageUnion, Sample


class Grader(ABC):
    """Base interface for graders."""

    def extract(
        self,
        trajectory: List[List[LettaMessageUnion]],
        context: Optional[Dict[str, Any]] = None,
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
        submission = self.extractor(trajectory, context or {})
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
        self, sample: Sample, trajectory: List[List[LettaMessageUnion]], context: Optional[Dict[str, Any]] = None
    ) -> Tuple[GradeResult, str]:
        """Grade a trajectory and return the result and extracted submission."""
        pass
