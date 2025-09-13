from enum import Enum
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress tracking during evaluation runs."""

    async def sample_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        """Called when a sample evaluation starts."""
        ...

    async def agent_loading(self, sample_id: int, model_name: Optional[str] = None) -> None:
        """Called when an agent is being loaded."""
        ...

    async def message_sending(self, sample_id: int, message_num: int, total_messages: int) -> None:
        """Called when sending messages to the agent."""
        ...

    async def grading_started(self, sample_id: int) -> None:
        """Called when grading of a sample begins."""
        ...

    async def sample_completed(self, sample_id: int, passed: bool, score: Optional[float] = None) -> None:
        """Called when a sample evaluation completes successfully."""
        ...

    async def sample_error(self, sample_id: int, error: str) -> None:
        """Called when a sample evaluation encounters an error."""
        ...


class TargetKind(str, Enum):
    AGENT = "agent"


class GraderKind(str, Enum):
    TOOL = "tool"
    RUBRIC = "rubric"


class MetricOp(str, Enum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"


class LLMProvider(str, Enum):
    OPENAI = "openai"
