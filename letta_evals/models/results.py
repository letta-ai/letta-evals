"""Per-sample result and primitive models.

Includes the message union type alias used throughout the codebase, the
target/grader output models (TargetResult, GradeResult, PerTurnGrade), the
per-sample primitives (Usage, Timing, Error) plus their aggregate companions
(TimingStats, ErrorSummary — kept here so the per-sample/aggregate pairs stay
co-located), and the per-sample SampleResult written to <model>.jsonl.
"""

from typing import Any, Dict, List, Optional, Union

from letta_client.types import AgentState, ToolReturnMessage
from letta_client.types.agents import (
    ApprovalRequestMessage,
    ApprovalResponseMessage,
    AssistantMessage,
    EventMessage,
    HiddenReasoningMessage,
    ReasoningMessage,
    SummaryMessage,
    SystemMessage,
    ToolCallMessage,
    UserMessage,
)
from pydantic import BaseModel, Field, field_validator

from letta_evals.models.sample import SampleId
from letta_evals.types import ErrorCategory

# Type alias for message union (replaces LettaMessageUnion from v0.x SDK)
LettaMessageUnion = Union[
    SystemMessage,
    UserMessage,
    ReasoningMessage,
    HiddenReasoningMessage,
    ToolCallMessage,
    ToolReturnMessage,
    AssistantMessage,
    ApprovalRequestMessage,
    ApprovalResponseMessage,
    SummaryMessage,
    EventMessage,
]


# Target / Grader result models


class TurnTokenData(BaseModel):
    """Token-level data for a single message in a turn.

    Used by training pipelines (e.g. GRPO) that need per-token IDs and
    log-probabilities from the generation model.
    """

    role: str = Field(description="Message role: assistant, tool, tool_call, tool_return, etc.")
    content: Optional[str] = Field(default=None, description="Text content of this message")
    input_ids: Optional[List[int]] = Field(
        default=None,
        description="Exact token IDs used as the model input/prefix for this generation",
    )
    output_ids: Optional[List[int]] = Field(
        default=None, description="Token IDs produced by the model for this message"
    )
    output_token_logprobs: Optional[List[Any]] = Field(
        default=None,
        description="Per-token log-probabilities. Each entry is either a float or a list whose first element is the logprob.",
    )


class TargetResult(BaseModel):
    """Result from running a target."""

    trajectory: List[List[LettaMessageUnion]] = Field(
        description="List of conversation turns, each containing Letta messages"
    )
    agent_id: str = Field(description="ID of the agent that generated this trajectory")
    model_handle: str = Field(description="Model handle used for this target")
    agent_usage: Optional[List[dict]] = Field(
        default=None, description="Usage statistics emitted by the agent during the run"
    )
    agent_state: Optional[AgentState] = Field(
        default=None, description="Agent state after running the target (includes memory blocks)"
    )
    token_data: Optional[List[TurnTokenData]] = Field(
        default=None,
        description=(
            "Token-level data (IDs + logprobs) for each message across all turns. "
            "Only populated when return_token_data=True is passed to the target."
        ),
    )


class PerTurnGrade(BaseModel):
    """Grade result for a single turn in per-turn evaluation.

    ``ground_truth`` is not stored here — look up ``sample.ground_truth[turn]``
    using the per-sample ``sample_id``.
    """

    turn: int = Field(description="Turn index (0-based)")
    score: float = Field(description="Score for this turn (0.0 to 1.0)")
    rationale: Optional[str] = Field(default=None, description="Explanation for this turn's grade")
    submission: str = Field(description="Extracted submission for this turn")


class GradeResult(BaseModel):
    """Grading result for one (sample, grader) pair."""

    score: float = Field(description="Numeric score between 0.0 and 1.0")
    rationale: Optional[str] = Field(default=None, description="Explanation of the grading decision")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional grading metadata")
    per_turn_grades: Optional[List[PerTurnGrade]] = Field(
        default=None, description="Per-turn grades for multi-turn evaluation (only populated for per-turn evaluations)"
    )

    @field_validator("score")
    def validate_score(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v


# Usage / Timing / Error primitives — used both per-sample and as aggregates.
# The aggregate companions (TimingStats, ErrorSummary) live here so each
# per-sample/aggregate pair stays co-located.


class Usage(BaseModel):
    """Token usage and cost. Used both per-sample and as a summed aggregate."""

    prompt_tokens: int = Field(default=0, description="Prompt tokens used")
    completion_tokens: int = Field(default=0, description="Completion tokens generated")
    cached_input_tokens: int = Field(default=0, description="Cached input tokens served from cache")
    cache_write_tokens: int = Field(default=0, description="Cache write tokens (Anthropic only)")
    reasoning_tokens: int = Field(default=0, description="Reasoning/thinking tokens generated")
    cost: Optional[float] = Field(default=None, description="Cost in dollars")


class Timing(BaseModel):
    """Per-sample timing (seconds)."""

    total: float = Field(description="Total wall time for this sample")
    target: float = Field(description="Wall time for target execution (agent creation + messages)")
    extraction: Optional[float] = Field(default=None, description="Wall time for extraction across all graders")
    per_grader: Optional[Dict[str, float]] = Field(default=None, description="Wall time per grader key")


class TimingStats(BaseModel):
    """Aggregate timing statistics across samples (seconds)."""

    mean_total: float = Field(description="Mean total wall time per sample")
    mean_target: float = Field(description="Mean target execution time per sample")
    mean_extraction: Optional[float] = Field(default=None, description="Mean extraction time per sample")
    p50_total: float = Field(description="Median total wall time per sample")
    p95_total: float = Field(description="95th percentile total wall time per sample")
    per_grader_mean: Optional[Dict[str, float]] = Field(default=None, description="Mean wall time per grader key")


class Error(BaseModel):
    """Structured error information for a failed sample."""

    category: ErrorCategory = Field(description="Category of the error")
    exception_type: str = Field(description="Python exception class name")
    message: str = Field(description="Full error message")


class ErrorSummary(BaseModel):
    """Summary of errors across an evaluation run (or a single model)."""

    total_errors: int = Field(description="Total number of samples that errored")
    by_category: Dict[str, int] = Field(default_factory=dict, description="Error count by category")
    by_exception_type: Dict[str, int] = Field(default_factory=dict, description="Error count by exception type")


class RewardOutput(BaseModel):
    """Composed reward for one sample.

    ``metadata`` is intentionally sparse: store only derived composer decisions
    that are not already recoverable from ``grades``, ``submissions``, or the
    sample.
    """

    score: float = Field(ge=0.0, le=1.0, description="Composed reward score between 0.0 and 1.0")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional derived reward metadata; do not duplicate raw grader scores"
    )


# Per-sample result. Lives in <model>.jsonl or <model>/run_<n>.jsonl.
#
# The model that produced this result is implicit in the file path; the
# Sample object is stored only once at the top level (suite.json), and looked
# up by ``sample_id``.


class SampleResult(BaseModel):
    """Result for a single sample evaluation."""

    sample_id: SampleId = Field(description="ID of the sample (look up the full Sample in suite.json)")
    agent_id: Optional[str] = Field(default=None, description="ID of the agent that generated this trajectory")
    trajectory: List[List[LettaMessageUnion]] = Field(description="Full conversation trajectory from the agent")
    submissions: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-grader extracted submissions (keyed by grader name)",
    )
    grades: Dict[str, GradeResult] = Field(
        default_factory=dict,
        description="Per-grader grading results (keyed by grader name)",
    )
    reward: Optional[RewardOutput] = Field(
        default=None,
        description="Composed per-sample reward. Null when the sample has a framework, target, grading, or reward error.",
    )
    usage: Optional[Usage] = Field(default=None, description="Token usage and cost for this sample")
    timing: Timing = Field(description="Wall-clock timing for this sample")
    error: Optional[Error] = Field(default=None, description="Structured error info if this sample failed")
    # Optional extras — omitted from serialized output when None.
    agent_usage: Optional[List[dict]] = Field(
        default=None, description="Raw usage statistics emitted by the agent during the run"
    )
    agent_state: Optional[AgentState] = Field(
        default=None,
        description=(
            "Agent state after running the target (includes memory blocks). "
            "Only populated when at least one grader requires agent_state."
        ),
    )
    token_data: Optional[List[TurnTokenData]] = Field(
        default=None,
        description=(
            "Per-token IDs and logprobs across all turns in the trajectory. "
            "Only populated when run_sample(return_token_data=True) is called."
        ),
    )
