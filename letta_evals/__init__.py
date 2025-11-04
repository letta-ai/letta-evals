"""Letta Evals Kit - Evaluation framework for Letta AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from letta_evals.decorators import agent_factory, extractor, grader, suite_setup
from letta_evals.graders import AgentJudgeGrader, Grader, RubricGrader, ToolGrader
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
from letta_evals.runner import Runner, run_suite
from letta_evals.targets import AbstractAgentTarget, LettaAgentTarget
from letta_evals.types import Aggregation, GateKind, GraderKind, LLMProvider, LogicalOp, MetricOp, TargetKind
from letta_evals.visualization.factory import ProgressStyle, create_progress_callback

try:
    __version__: str = _pkg_version("letta-evals")
except PackageNotFoundError:
    __version__ = "0.7.0"

__all__ = [
    "Sample",
    "SuiteSpec",
    "GateSpec",
    "TargetSpec",
    "GraderSpec",
    "run_suite",
    "Runner",
    "AbstractAgentTarget",
    "LettaAgentTarget",
    "Grader",
    "ToolGrader",
    "RubricGrader",
    "AgentJudgeGrader",
    "RunnerResult",
    "GradeResult",
    "SampleResult",
    "TargetResult",
    "Metrics",
    "ModelMetrics",
    "MetricAggregate",
    "RunStatistics",
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
