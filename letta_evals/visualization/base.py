from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


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
        self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> None:
        """Called when a sample evaluation starts."""
        ...

    async def agent_loading(
        self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None, from_cache: bool = False
    ) -> None:
        """Called when an agent is being loaded."""
        pass

    async def message_sending(
        self,
        sample_id: int,
        message_num: int,
        total_messages: int,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Called when sending messages to the agent."""
        pass

    async def grading_started(
        self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> None:
        """Called when grading of a sample begins."""
        pass

    async def turn_graded(
        self,
        sample_id: int,
        turn_num: int,
        total_turns: int,
        turn_score: float,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Called when a single turn is graded in per-turn evaluation mode."""
        pass

    @abstractmethod
    async def sample_completed(
        self,
        sample_id: int,
        agent_id: Optional[str] = None,
        score: Optional[float] = None,
        model_name: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
        rationale: Optional[str] = None,
        metric_rationales: Optional[Dict[str, str]] = None,
    ) -> None:
        """Called when a sample evaluation completes successfully."""
        ...

    @abstractmethod
    async def sample_error(
        self, sample_id: int, error: str, agent_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> None:
        """Called when a sample evaluation encounters an error."""
        ...
