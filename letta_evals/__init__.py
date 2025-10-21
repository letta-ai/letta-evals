"""Letta Evals Kit - Evaluation framework for Letta AI agents."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from letta_evals.graders.rubric import RubricGrader
from letta_evals.graders.tool import ToolGrader
from letta_evals.models import GateSpec, RunnerResult, Sample, SuiteSpec
from letta_evals.runner import Runner, run_suite
from letta_evals.targets.agent import AgentTarget

try:
    __version__: str = _pkg_version("letta-evals")
except PackageNotFoundError:
    __version__ = "0.5.0"

__all__ = [
    "Sample",
    "SuiteSpec",
    "RunnerResult",
    "GateSpec",
    "Runner",
    "run_suite",
    "AgentTarget",
    "ToolGrader",
    "RubricGrader",
]
