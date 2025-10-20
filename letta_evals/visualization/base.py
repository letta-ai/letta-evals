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

    def stop(self) -> None:
        """Optional lifecycle: stop the progress UI (if any)."""
        pass

    @abstractmethod
    async def sample_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        """Called when a sample evaluation starts."""
        ...

    async def agent_loading(self, sample_id: int, model_name: Optional[str] = None, from_cache: bool = False) -> None:
        """Called when an agent is being loaded."""
        pass

    async def message_sending(
        self, sample_id: int, message_num: int, total_messages: int, model_name: Optional[str] = None
    ) -> None:
        """Called when sending messages to the agent."""
        pass

    async def grading_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        """Called when grading of a sample begins."""
        pass

    @abstractmethod
    async def sample_completed(
        self,
        sample_id: int,
        passed: bool,
        score: Optional[float] = None,
        model_name: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
        metric_pass: Optional[Dict[str, bool]] = None,
        rationale: Optional[str] = None,
        metric_rationales: Optional[Dict[str, str]] = None,
    ) -> None:
        """Called when a sample evaluation completes successfully."""
        ...

    @abstractmethod
    async def sample_error(self, sample_id: int, error: str, model_name: Optional[str] = None) -> None:
        """Called when a sample evaluation encounters an error."""
        ...
