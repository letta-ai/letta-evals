"""Pydantic models for letta-evals.

This package re-exports the full public surface so existing imports of the
form ``from letta_evals.models import X`` continue to work. The models are
grouped into focused submodules:

- :mod:`letta_evals.models.sample` — :class:`Sample` (dataset input)
- :mod:`letta_evals.models.specs` — config: target / grader / reward / suite
  specs
- :mod:`letta_evals.models.results` — :data:`LettaMessageUnion`, per-sample
  result models (:class:`TargetResult`, :class:`GradeResult`,
  :class:`RewardOutput`, :class:`SampleResult`) and shared primitives
  (:class:`Usage`, :class:`Timing`/:class:`TimingStats`,
  :class:`Error`/:class:`ErrorSummary`)
- :mod:`letta_evals.models.summaries` — aggregate summary models and the
  in-memory :class:`RunnerResult`/:class:`ModelRun` containers
"""

from letta_client.types import AgentState

from letta_evals.models.results import (
    Error,
    ErrorSummary,
    GradeResult,
    LettaMessageUnion,
    PerTurnGrade,
    RewardOutput,
    SampleResult,
    TargetResult,
    Timing,
    TimingStats,
    TurnTokenData,
    Usage,
)
from letta_evals.models.sample import Sample, SampleId
from letta_evals.models.specs import (
    BaseGraderSpec,
    CustomRewardSpec,
    GraderSpec,
    LettaCodeTargetSpec,
    MetricRewardSpec,
    ModalSandboxSpec,
    ModelJudgeGraderSpec,
    RewardSpec,
    SandboxSpec,
    SuiteSpec,
    ToolGraderSpec,
)
from letta_evals.models.summaries import (
    ModelRun,
    ModelSummary,
    PerRunSummary,
    RunnerResult,
    Summary,
)

__all__ = [
    "AgentState",
    "Sample",
    "SampleId",
    "LettaCodeTargetSpec",
    "BaseGraderSpec",
    "ToolGraderSpec",
    "ModelJudgeGraderSpec",
    "GraderSpec",
    "MetricRewardSpec",
    "CustomRewardSpec",
    "RewardSpec",
    "ModalSandboxSpec",
    "SandboxSpec",
    "SuiteSpec",
    "LettaMessageUnion",
    "TurnTokenData",
    "TargetResult",
    "PerTurnGrade",
    "GradeResult",
    "RewardOutput",
    "Usage",
    "Timing",
    "TimingStats",
    "Error",
    "ErrorSummary",
    "SampleResult",
    "PerRunSummary",
    "ModelSummary",
    "Summary",
    "ModelRun",
    "RunnerResult",
]
