"""Metrics aggregation for evaluation results.

Pure functions that turn ``List[SampleResult]`` (per-run, single-model) into
``ModelSummary`` and ``PerRunSummary``. Extracted from Runner to make them
independently testable.
"""

import statistics as _statistics
from collections import Counter
from typing import Dict, List, Optional

from letta_evals.models import (
    Error,
    ErrorSummary,
    ModelSummary,
    PerRunSummary,
    SampleResult,
    Timing,
    TimingStats,
    Usage,
)


def aggregate_usage(results: List[SampleResult]) -> Usage:
    """Sum usage across samples. Returns a Usage with all zeros if nothing.

    Note: ``cost`` is summed only over samples that report a cost; if no
    sample has a cost, ``cost`` is left ``None``.
    """
    prompt = 0
    completion = 0
    cached = 0
    cache_write = 0
    reasoning = 0
    costs: List[float] = []

    for r in results:
        if r.usage is None:
            continue
        prompt += r.usage.prompt_tokens
        completion += r.usage.completion_tokens
        cached += r.usage.cached_input_tokens
        cache_write += r.usage.cache_write_tokens
        reasoning += r.usage.reasoning_tokens
        if r.usage.cost is not None:
            costs.append(r.usage.cost)

    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_input_tokens=cached,
        cache_write_tokens=cache_write,
        reasoning_tokens=reasoning,
        cost=sum(costs) if costs else None,
    )


def aggregate_timing(results: List[SampleResult]) -> Optional[TimingStats]:
    """Aggregate per-sample timing into mean/p50/p95 stats. Returns None if empty."""
    timings: List[Timing] = [r.timing for r in results if r.timing is not None]
    if not timings:
        return None

    total_times = [t.total for t in timings]
    target_times = [t.target for t in timings]
    extraction_times = [t.extraction for t in timings if t.extraction is not None]

    n = len(total_times)
    sorted_totals = sorted(total_times)

    grader_keys: set = set()
    for t in timings:
        if t.per_grader:
            grader_keys.update(t.per_grader.keys())
    per_grader_mean: Dict[str, float] = {}
    for k in sorted(grader_keys):
        vals = [t.per_grader[k] for t in timings if t.per_grader and k in t.per_grader]
        per_grader_mean[k] = sum(vals) / len(vals) if vals else 0.0

    return TimingStats(
        mean_total=sum(total_times) / n,
        mean_target=sum(target_times) / len(target_times) if target_times else 0.0,
        mean_extraction=(sum(extraction_times) / len(extraction_times)) if extraction_times else None,
        p50_total=sorted_totals[n // 2],
        p95_total=sorted_totals[min(int(n * 0.95), n - 1)],
        per_grader_mean=per_grader_mean if per_grader_mean else None,
    )


def aggregate_errors(results: List[SampleResult]) -> Optional[ErrorSummary]:
    """Summarize errors. Returns None when there are no errors."""
    errors: List[Error] = [r.error for r in results if r.error is not None]
    if not errors:
        return None
    return ErrorSummary(
        total_errors=len(errors),
        by_category=dict(Counter(e.category.value for e in errors)),
        by_exception_type=dict(Counter(e.exception_type for e in errors)),
    )



def _per_metric_average(results: List[SampleResult], grader_keys: List[str]) -> Dict[str, float]:
    """Mean score per grader across attempted (non-error) samples, 0-1 scale."""
    attempted = [r for r in results if r.error is None]
    out: Dict[str, float] = {}
    for key in grader_keys:
        scores = [r.grades[key].score for r in attempted if key in r.grades]
        out[key] = sum(scores) / len(scores) if scores else 0.0
    return out


def _reward_average(results: List[SampleResult]) -> float:
    """Mean composed reward across successful samples."""
    rewards = [r.reward.score for r in results if r.error is None and r.reward is not None]
    return sum(rewards) / len(rewards) if rewards else 0.0



def summarize_model(
    *,
    model: str,
    results: List[SampleResult],
    grader_keys: List[str],
) -> ModelSummary:
    """Build a ModelSummary from a single run's worth of results for one model.

    For multi-run aggregation, build one PerRunSummary per run with
    ``summarize_run`` and then call ``summarize_runs``.
    """
    n_total = len(results)
    n_attempted = sum(1 for r in results if r.error is None)
    per_metric = _per_metric_average(results, grader_keys)
    reward = _reward_average(results)
    usage = aggregate_usage(results)
    timing = aggregate_timing(results) or TimingStats(
        mean_total=0.0, mean_target=0.0, mean_extraction=None, p50_total=0.0, p95_total=0.0
    )
    errors = aggregate_errors(results)

    return ModelSummary(
        model=model,
        n_total=n_total,
        n_attempted=n_attempted,
        reward=reward,
        per_metric=per_metric,
        usage=usage,
        timing=timing,
        errors=errors,
    )


def summarize_run(
    *,
    run: int,
    results: List[SampleResult],
    grader_keys: List[str],
) -> PerRunSummary:
    """Build a per-run summary for one model and one run."""
    per_metric = _per_metric_average(results, grader_keys)
    reward = _reward_average(results)
    usage = aggregate_usage(results)
    timing = aggregate_timing(results) or TimingStats(
        mean_total=0.0, mean_target=0.0, mean_extraction=None, p50_total=0.0, p95_total=0.0
    )
    n_errors = sum(1 for r in results if r.error is not None)
    return PerRunSummary(
        run=run,
        reward=reward,
        per_metric=per_metric,
        usage=usage,
        timing=timing,
        n_errors=n_errors,
    )


def summarize_runs(
    *,
    model: str,
    per_run_results: List[List[SampleResult]],
    grader_keys: List[str],
) -> ModelSummary:
    """Build a ModelSummary aggregating across multiple runs for one model.

    Populates ``reward_std``, ``per_metric_std``, and ``runs`` on the returned
    summary. Per-run usage is summed; per-run timing stats are averaged across
    runs (mean of means/percentiles).
    """
    if not per_run_results:
        raise ValueError("summarize_runs requires at least one run")

    run_summaries: List[PerRunSummary] = [
        summarize_run(run=i + 1, results=run_results, grader_keys=grader_keys)
        for i, run_results in enumerate(per_run_results)
    ]

    n_total = sum(len(r) for r in per_run_results) // len(per_run_results)  # per-run sample count
    n_attempted = sum(sum(1 for r in run_results if r.error is None) for run_results in per_run_results)

    rewards = [rs.reward for rs in run_summaries]
    reward_mean = _statistics.mean(rewards)
    reward_std = _statistics.stdev(rewards) if len(rewards) > 1 else 0.0

    per_metric_mean: Dict[str, float] = {}
    per_metric_std: Dict[str, float] = {}
    for key in grader_keys:
        vals = [rs.per_metric.get(key, 0.0) for rs in run_summaries]
        per_metric_mean[key] = _statistics.mean(vals) if vals else 0.0
        per_metric_std[key] = _statistics.stdev(vals) if len(vals) > 1 else 0.0

    # Sum usage across all runs.
    all_results: List[SampleResult] = [r for run in per_run_results for r in run]
    usage = aggregate_usage(all_results)

    # Aggregate timing across runs: average each timing field across run-level means.
    run_timings = [rs.timing for rs in run_summaries]
    timing = TimingStats(
        mean_total=_statistics.mean(t.mean_total for t in run_timings),
        mean_target=_statistics.mean(t.mean_target for t in run_timings),
        mean_extraction=(
            _statistics.mean(t.mean_extraction for t in run_timings if t.mean_extraction is not None)
            if any(t.mean_extraction is not None for t in run_timings)
            else None
        ),
        p50_total=_statistics.mean(t.p50_total for t in run_timings),
        p95_total=_statistics.mean(t.p95_total for t in run_timings),
        per_grader_mean=_average_per_grader([t.per_grader_mean for t in run_timings]),
    )

    errors = aggregate_errors(all_results)

    return ModelSummary(
        model=model,
        n_total=n_total,
        n_attempted=n_attempted,
        reward=reward_mean,
        per_metric=per_metric_mean,
        usage=usage,
        timing=timing,
        errors=errors,
        reward_std=reward_std,
        per_metric_std=per_metric_std,
        runs=run_summaries,
    )


def _average_per_grader(per_grader_dicts: List[Optional[Dict[str, float]]]) -> Optional[Dict[str, float]]:
    """Average per-grader mean times across runs."""
    keys: set = set()
    for d in per_grader_dicts:
        if d:
            keys.update(d.keys())
    if not keys:
        return None
    out: Dict[str, float] = {}
    for k in sorted(keys):
        vals = [d[k] for d in per_grader_dicts if d and k in d]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out
