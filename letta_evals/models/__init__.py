"""Pydantic models for letta-evals.

This package re-exports the full public surface so existing imports of the
form ``from letta_evals.models import X`` continue to work. The models are
grouped into focused submodules:

- :mod:`letta_evals.models.sample` — :class:`Sample` (dataset input)
- :mod:`letta_evals.models.specs` — config: target / grader / gate / suite
  specs plus gate helpers
- :mod:`letta_evals.models.results` — :data:`LettaMessageUnion`, per-sample
  result models (:class:`TargetResult`, :class:`GradeResult`,
  :class:`SampleResult`) and shared primitives (:class:`Usage`,
  :class:`Timing`/:class:`TimingStats`, :class:`Error`/:class:`ErrorSummary`)
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
    SampleResult,
    TargetResult,
    Timing,
    TimingStats,
    TurnTokenData,
    Usage,
)
from letta_evals.models.sample import Sample, SampleId
from letta_evals.models.specs import (
    DEFAULT_MODAL_IMAGE,
    BaseGraderSpec,
    BaseTargetSpec,
    GateSpec,
    GraderSpec,
    LettaAgentTargetSpec,
    LettaCodeTargetSpec,
    LettaJudgeGraderSpec,
    LogicalGateSpec,
    ModalSandboxSpec,
    ModelJudgeGraderSpec,
    SandboxSpec,
    SimpleCondition,
    SimpleGateSpec,
    SuiteSpec,
    TargetSpec,
    ToolGraderSpec,
    WeightedAverageGateSpec,
    _compare,
    compute_gate_score,
    normalize_weights,
)
from letta_evals.models.summaries import (
    ModelRun,
    ModelSummary,
    PerRunSummary,
    RunnerResult,
    Summary,
)

__all__ = [
    # re-exports from letta_client
    "AgentState",
    # sample
    "Sample",
    "SampleId",
    # specs — targets
    "BaseTargetSpec",
    "LettaAgentTargetSpec",
    "LettaCodeTargetSpec",
    "TargetSpec",
    # specs — graders
    "BaseGraderSpec",
    "ToolGraderSpec",
    "ModelJudgeGraderSpec",
    "LettaJudgeGraderSpec",
    "GraderSpec",
    # specs — gates
    "SimpleCondition",
    "SimpleGateSpec",
    "WeightedAverageGateSpec",
    "LogicalGateSpec",
    "GateSpec",
    # specs — sandbox
    "ModalSandboxSpec",
    "SandboxSpec",
    "DEFAULT_MODAL_IMAGE",
    # specs — suite
    "SuiteSpec",
    # specs — gate helpers
    "compute_gate_score",
    "normalize_weights",
    "_compare",
    # results — messages
    "LettaMessageUnion",
    # results — target / grader output
    "TurnTokenData",
    "TargetResult",
    "PerTurnGrade",
    "GradeResult",
    # results — per-sample primitives + aggregate companions
    "Usage",
    "Timing",
    "TimingStats",
    "Error",
    "ErrorSummary",
    # results — per-sample record
    "SampleResult",
    # summaries — aggregate
    "PerRunSummary",
    "ModelSummary",
    "Summary",
    # summaries — in-memory holders
    "ModelRun",
    "RunnerResult",
]
