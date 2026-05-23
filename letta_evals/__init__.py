"""Letta Evals Kit - Evaluation framework for Letta AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from letta_evals.decorators import agent_factory, extractor, grader, suite_setup
from letta_evals.graders import AgentJudgeGrader, Grader, RubricGrader, ToolGrader
from letta_evals.models import (
    AgentState,
    Error,
    ErrorSummary,
    GateSpec,
    GradeResult,
    GraderSpec,
    LettaMessageUnion,
    ModelRun,
    ModelSummary,
    PerRunSummary,
    PerTurnGrade,
    RunnerResult,
    Sample,
    SampleResult,
    SuiteSpec,
    Summary,
    TargetResult,
    TargetSpec,
    Timing,
    TimingStats,
    TurnTokenData,
    Usage,
)
from letta_evals.runner import Runner, run_suite
from letta_evals.targets import AbstractAgentTarget, LettaAgentTarget
from letta_evals.types import (
    Aggregation,
    ErrorCategory,
    GateKind,
    GraderKind,
    LLMProvider,
    LogicalOp,
    MetricOp,
    TargetKind,
)
from letta_evals.visualization.factory import ProgressStyle, create_progress_callback

try:
    __version__: str = _pkg_version("letta-evals")
except PackageNotFoundError:
    __version__ = "0.18.0"

__all__ = [
    "AbstractAgentTarget",
    "AgentJudgeGrader",
    "AgentState",
    "Aggregation",
    "Error",
    "ErrorCategory",
    "ErrorSummary",
    "GateKind",
    "GateSpec",
    "GradeResult",
    "Grader",
    "GraderKind",
    "GraderSpec",
    "LettaAgentTarget",
    "LettaMessageUnion",
    "LLMProvider",
    "LogicalOp",
    "MetricOp",
    "ModelRun",
    "ModelSummary",
    "PerRunSummary",
    "PerTurnGrade",
    "ProgressStyle",
    "RubricGrader",
    "Runner",
    "RunnerResult",
    "Sample",
    "SampleResult",
    "Summary",
    "SuiteSpec",
    "TargetKind",
    "TargetResult",
    "TargetSpec",
    "Timing",
    "TimingStats",
    "ToolGrader",
    "TurnTokenData",
    "Usage",
    "agent_factory",
    "create_progress_callback",
    "extractor",
    "grader",
    "run_suite",
    "suite_setup",
]
