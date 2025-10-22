from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

from rich.console import Console

from letta_evals.models import ModelJudgeGraderSpec, SuiteSpec
from letta_evals.visualization.base import ProgressCallback
from letta_evals.visualization.noop_progress import NoOpProgress
from letta_evals.visualization.rich_progress import DisplayMode, EvalProgress
from letta_evals.visualization.simple_progress import SimpleProgress


class ProgressStyle(str, Enum):
    """Built-in progress verbosity levels."""

    NONE = "none"
    SIMPLE = "simple"
    RICH = "rich"


def create_progress_callback(
    style: ProgressStyle,
    suite: SuiteSpec,
    total_evaluations: int,
    *,
    console: Optional[Console] = None,
    max_concurrent: int = 15,
    cached_mode: bool = False,
    metric_labels: Optional[Dict[str, str]] = None,
) -> ProgressCallback:
    """Factory for built-in progress callbacks.

    - RICH: full Rich-based UI from visualization.progress
    - SIMPLE: single-line updates suitable for logs
    - NONE: silent
    """
    if style == ProgressStyle.NONE:
        return NoOpProgress()

    if style == ProgressStyle.SIMPLE:
        return SimpleProgress(suite_name=suite.name, total_samples=total_evaluations, console=console)

    # RICH (default for CLI)
    # Determine grader kind label for header
    if suite.graders and len(suite.graders) == 1:
        only_kind = next(iter(suite.graders.values())).kind.value
        grader_kind_label = only_kind
    else:
        grader_kind_label = "multi"

    # choose model if any grader is model_judge
    rubric_model = None
    if suite.graders:
        for _, gspec in suite.graders.items():
            if isinstance(gspec, ModelJudgeGraderSpec):
                rubric_model = gspec.model
                break

    progress = EvalProgress(
        suite_name=suite.name,
        total_samples=total_evaluations,
        target_kind=suite.target.kind.value,
        grader_kind=grader_kind_label,
        rubric_model=rubric_model,
        max_concurrent=max_concurrent,
        display_mode=DisplayMode.DETAILED,
        console=console,
        show_samples=True,
        cached_mode=cached_mode,
        metric_labels=metric_labels,
    )
    return progress


__all__ = ["ProgressStyle", "create_progress_callback"]
