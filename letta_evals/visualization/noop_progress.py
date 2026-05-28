from __future__ import annotations

from typing import Dict, Optional

from letta_evals.models.sample import SampleId
from letta_evals.visualization.base import ProgressCallback


class NoOpProgress(ProgressCallback):
    """No-output progress callback (silent)."""

    async def sample_started(
        self, sample_id: SampleId, agent_id: Optional[str] = None, model_handle: Optional[str] = None
    ) -> None:
        pass

    async def sample_completed(
        self,
        sample_id: SampleId,
        agent_id: Optional[str] = None,
        score: Optional[float] = None,
        target_cost: Optional[float] = None,
        model_handle: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
        rationale: Optional[str] = None,
        metric_rationales: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    async def sample_error(
        self,
        sample_id: SampleId,
        error: str,
        agent_id: Optional[str] = None,
        model_handle: Optional[str] = None,
        target_cost: Optional[float] = None,
    ) -> None:
        pass
