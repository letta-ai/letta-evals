"""Letta Evals Kit - Evaluation framework for Letta AI agents."""

from letta_evals.graders.rubric import RubricGrader
from letta_evals.graders.tool import ToolGrader
from letta_evals.models import GateSpec, RunnerResult, Sample, SuiteSpec
from letta_evals.runner import Runner, run_suite
from letta_evals.targets.agent import AgentTarget

__version__ = "0.1.0b1"

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
