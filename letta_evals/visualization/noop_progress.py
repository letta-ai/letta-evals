from __future__ import annotations

from typing import Optional

from letta_evals.models import SampleResult
from letta_evals.models.sample import SampleId
from letta_evals.visualization.base import ProgressCallback


class NoOpProgress(ProgressCallback):
    """No-output progress callback (silent)."""

    async def sample_started(
        self, sample_id: SampleId, agent_id: Optional[str] = None, model_handle: Optional[str] = None
    ) -> None:
        pass

    async def sample_completed(self, result: SampleResult, model_handle: Optional[str] = None) -> None:
        pass

    async def sample_error(self, result: SampleResult, model_handle: Optional[str] = None) -> None:
        pass
