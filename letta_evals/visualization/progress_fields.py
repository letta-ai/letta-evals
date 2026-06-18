from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from letta_evals.models import SampleResult


@dataclass(frozen=True)
class SampleProgressFields:
    """Display fields derived from a completed SampleResult."""

    reward: float
    target_cost: Optional[float]
    rationale: Optional[str]
    metric_scores: dict[str, float]
    metric_rationales: dict[str, str]


def sample_progress_fields(result: SampleResult) -> SampleProgressFields:
    """Build visualization fields from the SampleResult source of truth."""
    metric_scores = {key: grade.score for key, grade in result.grades.items()}
    metric_rationales = {key: (grade.rationale or "") for key, grade in result.grades.items()}
    only_grade = next(iter(result.grades.values())) if len(result.grades) == 1 else None
    cost = result.usage.cost if result.usage and result.usage.cost and result.usage.cost > 0 else None

    return SampleProgressFields(
        reward=result.reward.score if result.reward is not None else 0.0,
        target_cost=cost,
        rationale=only_grade.rationale if only_grade else None,
        metric_scores=metric_scores,
        metric_rationales=metric_rationales,
    )
