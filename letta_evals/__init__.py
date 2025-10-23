"""Letta Evals Kit - Evaluation framework for Letta AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# decorators
from letta_evals.decorators import agent_factory, extractor, grader, suite_setup

# grader classes
from letta_evals.graders import AgentJudgeGrader, Grader, RubricGrader, ToolGrader

# core config models
# result models
from letta_evals.models import (
    GateSpec,
    GradeResult,
    GraderSpec,
    MetricAggregate,
    Metrics,
    ModelMetrics,
    RunnerResult,
    RunStatistics,
    Sample,
    SampleResult,
    SuiteSpec,
    TargetResult,
    TargetSpec,
)

# runner
from letta_evals.runner import Runner, run_suite

# target classes
from letta_evals.targets import AgentTarget, Target

# types/enums
from letta_evals.types import GateMetric, GraderKind, LLMProvider, MetricOp, TargetKind

# visualization
from letta_evals.visualization.factory import ProgressStyle, create_progress_callback

try:
    __version__: str = _pkg_version("letta-evals")
except PackageNotFoundError:
    __version__ = "0.4.1"

__all__ = [
    # core config models
    "Sample",
    "SuiteSpec",
    "GateSpec",
    "TargetSpec",
    "GraderSpec",
    # runner
    "run_suite",
    "Runner",
    # target classes
    "Target",
    "AgentTarget",
    # grader classes
    "Grader",
    "ToolGrader",
    "RubricGrader",
    "AgentJudgeGrader",
    # result models
    "RunnerResult",
    "GradeResult",
    "SampleResult",
    "TargetResult",
    "Metrics",
    "ModelMetrics",
    "MetricAggregate",
    "RunStatistics",
    # types/enums
    "GraderKind",
    "TargetKind",
    "MetricOp",
    "GateMetric",
    "LLMProvider",
    # decorators
    "grader",
    "extractor",
    "agent_factory",
    "suite_setup",
    # visualization
    "ProgressStyle",
    "create_progress_callback",
]
