"""Letta Evals Kit - Evaluation framework for Letta AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from letta_evals.decorators import agent_factory, extractor, grader, reward_composer, suite_setup
from letta_evals.graders import Grader, RubricGrader, ToolGrader
from letta_evals.models import (
    AgentState,
    CustomRewardSpec,
    Error,
    ErrorSummary,
    GradeResult,
    GraderSpec,
    LettaMessageUnion,
    MetricRewardSpec,
    ModelRun,
    ModelSummary,
    PerRunSummary,
    PerTurnGrade,
    RewardOutput,
    RewardSpec,
    RunnerResult,
    Sample,
    SampleResult,
    SuiteSpec,
    Summary,
    TargetResult,
    Timing,
    TimingStats,
    TurnTokenData,
    Usage,
)
from letta_evals.rewards import RewardContext
from letta_evals.runner import Runner, run_suite
from letta_evals.targets import LettaCodeTarget, TargetError
from letta_evals.types import (
    ErrorCategory,
    GraderKind,
    LLMProvider,
    RewardKind,
)
from letta_evals.visualization.factory import ProgressStyle, create_progress_callback

try:
    __version__: str = _pkg_version("letta-evals")
except PackageNotFoundError:
    __version__ = "0.24.0"

__all__ = [
    "AgentState",
    "CustomRewardSpec",
    "Error",
    "ErrorCategory",
    "ErrorSummary",
    "GradeResult",
    "Grader",
    "GraderKind",
    "GraderSpec",
    "LettaCodeTarget",
    "LettaMessageUnion",
    "LLMProvider",
    "MetricRewardSpec",
    "ModelRun",
    "ModelSummary",
    "PerRunSummary",
    "PerTurnGrade",
    "ProgressStyle",
    "RewardContext",
    "RewardKind",
    "RewardOutput",
    "RewardSpec",
    "RubricGrader",
    "Runner",
    "RunnerResult",
    "Sample",
    "SampleResult",
    "Summary",
    "SuiteSpec",
    "TargetError",
    "TargetResult",
    "Timing",
    "TimingStats",
    "ToolGrader",
    "TurnTokenData",
    "Usage",
    "agent_factory",
    "create_progress_callback",
    "extractor",
    "grader",
    "reward_composer",
    "run_suite",
    "suite_setup",
]
