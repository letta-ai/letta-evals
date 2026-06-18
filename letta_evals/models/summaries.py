"""Aggregate summary and in-memory run holders.

``Summary`` / ``ModelSummary`` / ``PerRunSummary`` describe the on-disk
summary.json layout produced by streaming. ``RunnerResult`` / ``ModelRun``
are the in-memory shapes returned by ``run_suite`` and consumed by the
visualization layer.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from letta_evals.models.results import ErrorSummary, SampleResult, TimingStats, Usage
from letta_evals.models.sample import Sample
from letta_evals.models.specs import SuiteSpec


class PerRunSummary(BaseModel):
    """Per-run summary for a single model (multi-run only)."""

    run: int = Field(description="Run index (1-based)")
    reward: float = Field(description="Mean reward for this run (0-1)")
    per_metric: Dict[str, float] = Field(description="Per-grader average score for this run (0-1)")
    usage: Usage = Field(description="Aggregate usage for this run")
    timing: TimingStats = Field(description="Aggregate timing for this run")
    n_errors: int = Field(default=0, description="Number of errored samples in this run")


class ModelSummary(BaseModel):
    """Summary metrics for one model across one or more runs."""

    model: str = Field(description="Model identifier")
    n_total: int = Field(description="Total samples scheduled (success + error) per run")
    n_attempted: int = Field(description="Samples completed without error (sum across runs)")
    reward: float = Field(description="Mean composed reward (0-1, mean across runs)")
    reward_std: Optional[float] = Field(
        default=None, description="Standard deviation of reward across runs (multi-run only)"
    )
    per_metric: Dict[str, float] = Field(description="Per-grader average score (0-1, mean across runs)")
    per_metric_std: Optional[Dict[str, float]] = Field(
        default=None,
        description="Standard deviation of per_metric values across runs (multi-run only)",
    )
    usage: Usage = Field(description="Aggregate usage (summed across runs)")
    timing: TimingStats = Field(description="Aggregate timing (averaged across runs)")
    errors: Optional[ErrorSummary] = Field(default=None, description="Error breakdown (only when errors > 0)")
    runs: Optional[List[PerRunSummary]] = Field(
        default=None,
        description="Per-run breakdown (only present in per-model summary.json, not top-level)",
    )


class Summary(BaseModel):
    """Top-level evaluation summary."""

    suite: str = Field(description="Name of the evaluation suite")
    models: List[ModelSummary] = Field(description="Per-model summary (one entry per model)")


# In-memory holders. ``RunnerResult`` is what ``run_suite`` returns and what
# the visualization layer consumes. Disk layout is owned by
# ``letta_evals.streaming``.


class ModelRun(BaseModel):
    """In-memory results for one model (one or more runs)."""

    model: str = Field(description="Model identifier")
    results: List[SampleResult] = Field(
        description="Sample results. For num_runs > 1 this is the last run; see ``runs`` for full history.",
    )
    runs: Optional[List[List[SampleResult]]] = Field(
        default=None,
        description="Per-run sample results, indexed as runs[run_idx][sample_idx] (only when num_runs > 1)",
    )
    summary: ModelSummary = Field(description="Aggregate summary for this model")


class RunnerResult(BaseModel):
    """Complete evaluation run result returned by ``run_suite``."""

    suite_spec: SuiteSpec = Field(description="The full suite configuration that produced this run")
    samples: List[Sample] = Field(description="The dataset, loaded once and shared across all model runs")
    runs: Dict[str, ModelRun] = Field(description="Per-model run data, keyed by model identifier")
    summary: Summary = Field(description="Top-level summary across all models")
