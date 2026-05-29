from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from letta_evals.models import SampleResult
from letta_evals.models.sample import SampleId


class ProgressCallback(ABC):
    """Abstract base class for progress tracking during evaluation runs.

    Subclasses must implement the core callback methods (sample_started, sample_completed,
    sample_error). Optional lifecycle and fine-grained hooks have default no-op implementations.
    """

    async def start(self) -> None:
        """Optional lifecycle: start the progress UI (if any)."""
        pass

    async def suite_completed(self, result) -> None:
        """Optional lifecycle: called when evaluation completes with final results.

        Args:
            result: RunnerResult object containing metrics, sample results, and config
        """
        pass

    def stop(self) -> None:
        """Optional lifecycle: stop the progress UI (if any)."""
        pass

    def reset(self) -> None:
        """Optional lifecycle: reset state for a new run (for multi-run scenarios)."""
        pass

    @abstractmethod
    async def sample_started(
        self, sample_id: SampleId, agent_id: Optional[str] = None, model_handle: Optional[str] = None
    ) -> None:
        """Called when a sample evaluation starts."""
        ...

    async def agent_created(
        self, sample_id: SampleId, agent_id: str, model_handle: Optional[str] = None, from_cache: bool = False
    ) -> None:
        """Called when an agent has been created/provisioned or resolved from cache.

        Fires as soon as agent_id is known, before messages are sent.
        """
        pass

    async def message_sending(
        self,
        sample_id: SampleId,
        message_num: int,
        total_messages: int,
        agent_id: Optional[str] = None,
        model_handle: Optional[str] = None,
    ) -> None:
        """Called when sending messages to the agent."""
        pass

    async def grading_started(
        self, sample_id: SampleId, agent_id: Optional[str] = None, model_handle: Optional[str] = None
    ) -> None:
        """Called when grading of a sample begins."""
        pass

    async def turn_graded(
        self,
        sample_id: SampleId,
        turn_num: int,
        total_turns: int,
        turn_score: float,
        grader_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        model_handle: Optional[str] = None,
    ) -> None:
        """Called when a single turn is graded in per-turn evaluation mode."""
        pass

    @abstractmethod
    async def sample_completed(self, result: SampleResult, model_handle: Optional[str] = None) -> None:
        """Called when a sample evaluation completes successfully."""
        ...

    @abstractmethod
    async def sample_error(self, result: SampleResult, model_handle: Optional[str] = None) -> None:
        """Called when a sample evaluation encounters an error."""
        ...
