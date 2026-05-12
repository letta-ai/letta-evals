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
    Summary,
    SuiteSpec,
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
    __version__ = "0.16.0"

__all__ = [
    # core domain
    "Sample",
    "SuiteSpec",
    "GateSpec",
    "TargetSpec",
    "GraderSpec",
    "TargetResult",
    "TurnTokenData",
    # per-sample
    "SampleResult",
    "GradeResult",
    "PerTurnGrade",
    # primitives
    "Usage",
    "Timing",
    "TimingStats",
    "Error",
    "ErrorSummary",
    # summaries
    "Summary",
    "ModelSummary",
    "ModelRun",
    "PerRunSummary",
    # runners / clients
    "RunnerResult",
    "Runner",
    "run_suite",
    "AbstractAgentTarget",
    "LettaAgentTarget",
    # graders
    "Grader",
    "ToolGrader",
    "RubricGrader",
    "AgentJudgeGrader",
    # misc
    "AgentState",
    "LettaMessageUnion",
    "ErrorCategory",
    "GraderKind",
    "TargetKind",
    "MetricOp",
    "Aggregation",
    "GateKind",
    "LogicalOp",
    "LLMProvider",
    "grader",
    "extractor",
    "agent_factory",
    "suite_setup",
    "ProgressStyle",
    "create_progress_callback",
]
