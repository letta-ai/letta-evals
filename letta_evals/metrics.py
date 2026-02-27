"""Metrics computation for evaluation results.

Pure functions that aggregate SampleResult lists into metrics models.
Extracted from Runner to make them independently testable.
"""

import statistics as _statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from letta_evals.models import (
    ErrorSummary,
    MetricAggregate,
    Metrics,
    ModelMetrics,
    RunStatistics,
    SampleResult,
    SuiteSpec,
    TimingMetrics,
    UsageMetrics,
)


def compute_usage_metrics(results: List[SampleResult]) -> Optional[UsageMetrics]:
    """Compute aggregate usage metrics from a list of sample results."""
    costs = [r.cost for r in results if r.cost is not None]
    total_cost = sum(costs) if costs else None

    total_prompt = sum(r.prompt_tokens for r in results if r.prompt_tokens is not None)
    total_completion = sum(r.completion_tokens for r in results if r.completion_tokens is not None)
    total_cached = sum(r.cached_input_tokens for r in results if r.cached_input_tokens is not None)
    total_cache_write = sum(r.cache_write_tokens for r in results if r.cache_write_tokens is not None)
    total_reasoning = sum(r.reasoning_tokens for r in results if r.reasoning_tokens is not None)

    if total_prompt > 0 or total_completion > 0 or (total_cost is not None and total_cost > 0):
        return UsageMetrics(
            total_prompt_tokens=total_prompt,
            total_completion_tokens=total_completion,
            total_cost=total_cost if total_cost and total_cost > 0 else None,
            total_cached_input_tokens=total_cached,
            total_cache_write_tokens=total_cache_write,
            total_reasoning_tokens=total_reasoning,
        )
    return None


def compute_error_summary(results: List[SampleResult]) -> Optional[ErrorSummary]:
    """Compute error summary from a list of sample results."""
    error_results = [r for r in results if r.error is not None]
    if not error_results:
        return None
    return ErrorSummary(
        total_errors=len(error_results),
        by_category=dict(Counter(r.error.category.value for r in error_results)),
        by_exception_type=dict(Counter(r.error.exception_type for r in error_results)),
        failed_sample_ids=sorted(r.sample.id for r in error_results),
    )


def compute_timing_metrics(results: List[SampleResult]) -> Optional[TimingMetrics]:
    """Compute aggregate timing statistics from a list of sample results."""
    total_times = [r.total_time for r in results if r.total_time is not None]
    if not total_times:
        return None

    target_times = [r.target_time for r in results if r.target_time is not None]
    extraction_times = [r.extraction_time for r in results if r.extraction_time is not None]

    n = len(total_times)
    sorted_totals = sorted(total_times)

    # Aggregate per-grader times
    grader_keys: set = set()
    for r in results:
        if r.per_grader_time:
            grader_keys.update(r.per_grader_time.keys())
    per_grader_mean: Dict[str, float] = {}
    for k in sorted(grader_keys):
        vals = [r.per_grader_time[k] for r in results if r.per_grader_time and k in r.per_grader_time]
        per_grader_mean[k] = sum(vals) / len(vals) if vals else 0.0

    return TimingMetrics(
        mean_total_seconds=sum(total_times) / n,
        mean_target_seconds=sum(target_times) / len(target_times) if target_times else 0.0,
        mean_extraction_seconds=sum(extraction_times) / len(extraction_times) if extraction_times else 0.0,
        p50_total_seconds=sorted_totals[n // 2],
        p95_total_seconds=sorted_totals[min(int(n * 0.95), n - 1)],
        per_grader_mean_seconds=per_grader_mean if per_grader_mean else None,
    )


def calculate_metrics(
    results: List[SampleResult],
    grader_keys: Optional[List[str]],
    has_multi_model: bool,
) -> Metrics:
    """Calculate aggregate metrics from results.

    Args:
        results: All sample results from the evaluation run.
        grader_keys: Ordered list of grader keys (from suite.graders), or None for single-grader.
        has_multi_model: Whether the eval used multiple model configs.

    Returns:
        Metrics with overall and optionally per-model breakdowns.
    """
    total = len(results)
    if total == 0:
        return Metrics(
            total=0,
            total_attempted=0,
            avg_score_attempted=0.0,
            avg_score_total=0.0,
            metrics={},
        )

    def is_success(r: SampleResult) -> bool:
        return r.error is None

    attempted = sum(1 for r in results if is_success(r))

    error_summary = compute_error_summary(results)

    # compute per-metric aggregates if multiple graders
    by_metric: Dict[str, MetricAggregate] = {}
    if grader_keys:
        for metric_key in grader_keys:
            m_scores = [
                r.grades[metric_key].score for r in results if is_success(r) and r.grades and metric_key in r.grades
            ]
            m_avg_attempted = sum(m_scores) / len(m_scores) if m_scores else 0.0
            m_avg_total = sum(m_scores) / len(results) if m_scores else 0.0
            # pass_rate is just avg score as percentage
            m_pass_rate = m_avg_attempted * 100.0
            by_metric[metric_key] = MetricAggregate(
                avg_score_attempted=m_avg_attempted,
                avg_score_total=m_avg_total,
                pass_rate=m_pass_rate,
            )

    metrics_dict: Dict[str, float] = {}
    if grader_keys:
        # use first grader for overall metrics
        first_key = grader_keys[0]
        for key, agg in by_metric.items():
            metrics_dict[key] = agg.pass_rate

        agg = by_metric.get(first_key) if first_key in by_metric else None
        avg_score_attempted = agg.avg_score_attempted if agg else 0.0
        avg_score_total = agg.avg_score_total if agg else 0.0
    else:
        scores = [r.grade.score for r in results if is_success(r)]
        avg_score_attempted = sum(scores) / len(scores) if scores else 0.0
        avg_score_total = sum(scores) / len(results) if scores else 0.0
        # for single grader case, use a default key
        default_key = "default"
        metrics_dict[default_key] = avg_score_attempted * 100.0

    usage_metrics = compute_usage_metrics(results)
    timing_metrics = compute_timing_metrics(results)

    per_model = None
    if has_multi_model:
        model_results: Dict[str, List[SampleResult]] = defaultdict(list)
        for result in results:
            model_results[result.model_name].append(result)

        per_model = []
        for model_name, m_results in sorted(model_results.items()):
            model_attempted = sum(1 for r in m_results if is_success(r))
            model_metrics_dict: Dict[str, float] = {}

            if grader_keys:
                # use first grader for overall model metrics
                first_key = grader_keys[0]
                # calculate avg score for each metric
                for metric_key in grader_keys:
                    metric_scores = [
                        r.grades[metric_key].score
                        for r in m_results
                        if is_success(r) and r.grades and metric_key in r.grades
                    ]
                    model_metrics_dict[metric_key] = (
                        (sum(metric_scores) / len(metric_scores)) * 100.0 if metric_scores else 0.0
                    )

                model_scores = [
                    r.grades[first_key].score for r in m_results if is_success(r) and r.grades and first_key in r.grades
                ]
            else:
                model_scores = [r.grade.score for r in m_results if is_success(r)]
                default_key = "default"
                model_metrics_dict[default_key] = (
                    (sum(model_scores) / len(model_scores)) * 100.0 if model_scores else 0.0
                )

            model_avg_attempted = sum(model_scores) / len(model_scores) if model_scores else 0.0
            model_avg_total = sum(model_scores) / len(m_results) if model_scores else 0.0

            model_usage_metrics = compute_usage_metrics(m_results)
            model_timing_metrics = compute_timing_metrics(m_results)
            model_error_summary = compute_error_summary(m_results)

            per_model.append(
                ModelMetrics(
                    model_name=model_name,
                    total=len(m_results),
                    total_attempted=model_attempted,
                    avg_score_attempted=model_avg_attempted,
                    avg_score_total=model_avg_total,
                    metrics=model_metrics_dict,
                    usage_metrics=model_usage_metrics,
                    timing_metrics=model_timing_metrics,
                    error_summary=model_error_summary,
                )
            )

    return Metrics(
        total=total,
        total_attempted=attempted,
        avg_score_attempted=avg_score_attempted,
        avg_score_total=avg_score_total,
        per_model=per_model,
        by_metric=by_metric if by_metric else None,
        metrics=metrics_dict,
        usage_metrics=usage_metrics,
        timing_metrics=timing_metrics,
        error_summary=error_summary,
    )


def calculate_run_statistics(all_metrics: List[Metrics], runs_passed: int, suite: SuiteSpec) -> RunStatistics:
    """Calculate aggregate statistics across multiple runs."""
    num_runs = len(all_metrics)

    avg_scores_attempted = [m.avg_score_attempted for m in all_metrics]
    avg_scores_total = [m.avg_score_total for m in all_metrics]

    mean_avg_score_attempted = _statistics.mean(avg_scores_attempted)
    std_avg_score_attempted = _statistics.stdev(avg_scores_attempted) if num_runs > 1 else 0.0

    mean_avg_score_total = _statistics.mean(avg_scores_total)
    std_avg_score_total = _statistics.stdev(avg_scores_total) if num_runs > 1 else 0.0

    mean_scores: Dict[str, float] = {}
    std_scores: Dict[str, float] = {}

    if suite.graders:
        for metric_key in suite.graders.keys():
            metric_values = []
            for m in all_metrics:
                if m.by_metric and metric_key in m.by_metric:
                    metric_values.append(m.by_metric[metric_key].avg_score_attempted)

            if metric_values:
                mean_scores[metric_key] = _statistics.mean(metric_values)
                std_scores[metric_key] = _statistics.stdev(metric_values) if len(metric_values) > 1 else 0.0

    return RunStatistics(
        num_runs=num_runs,
        runs_passed=runs_passed,
        mean_avg_score_attempted=mean_avg_score_attempted,
        std_avg_score_attempted=std_avg_score_attempted,
        mean_avg_score_total=mean_avg_score_total,
        std_avg_score_total=std_avg_score_total,
        mean_scores=mean_scores,
        std_scores=std_scores,
        individual_run_metrics=all_metrics,
    )
