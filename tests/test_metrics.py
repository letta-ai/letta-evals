"""Unit tests for letta_evals.metrics module."""

import pytest

from letta_evals.metrics import (
    calculate_metrics,
    compute_error_summary,
    compute_timing_metrics,
    compute_usage_metrics,
)
from letta_evals.models import ErrorInfo, GradeResult, Sample, SampleResult
from letta_evals.types import ErrorCategory


def _make_sample(id: int = 0) -> Sample:
    return Sample(id=id, input="test input", ground_truth="test answer")


_UNSET = object()


def _make_result(
    sample_id: int = 0,
    score: float = 1.0,
    model_name: str = "gpt-4o",
    cost: float | None = 0.01,
    prompt_tokens: int | None = 100,
    completion_tokens: int | None = 50,
    total_time: float | None = 5.0,
    target_time: float | None = 4.5,
    extraction_time: float | None = 0.001,
    per_grader_time: dict | None | object = _UNSET,
    error: ErrorInfo | None = None,
    grades: dict | None = None,
) -> SampleResult:
    sample = _make_sample(sample_id)
    grade = GradeResult(score=score, rationale="test rationale")
    return SampleResult(
        sample=sample,
        submission="test submission",
        trajectory=[],
        grade=grade,
        grades=grades,
        model_name=model_name,
        cost=cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_time=total_time,
        target_time=target_time,
        extraction_time=extraction_time,
        per_grader_time={"grader_a": 0.01} if per_grader_time is _UNSET else per_grader_time,
        error=error,
    )


# ── compute_usage_metrics ──


class TestComputeUsageMetrics:
    def test_basic(self):
        results = [_make_result(cost=0.01, prompt_tokens=100, completion_tokens=50)]
        usage = compute_usage_metrics(results)
        assert usage is not None
        assert usage.total_prompt_tokens == 100
        assert usage.total_completion_tokens == 50
        assert usage.total_cost == 0.01

    def test_multiple_results_sum(self):
        results = [
            _make_result(cost=0.01, prompt_tokens=100, completion_tokens=50),
            _make_result(cost=0.02, prompt_tokens=200, completion_tokens=100),
        ]
        usage = compute_usage_metrics(results)
        assert usage is not None
        assert usage.total_prompt_tokens == 300
        assert usage.total_completion_tokens == 150
        assert usage.total_cost == pytest.approx(0.03)

    def test_none_tokens_excluded(self):
        results = [
            _make_result(prompt_tokens=100, completion_tokens=None, cost=None),
        ]
        usage = compute_usage_metrics(results)
        assert usage is not None
        assert usage.total_prompt_tokens == 100
        assert usage.total_completion_tokens == 0
        assert usage.total_cost is None

    def test_all_none_returns_none(self):
        results = [
            _make_result(prompt_tokens=None, completion_tokens=None, cost=None),
        ]
        usage = compute_usage_metrics(results)
        assert usage is None

    def test_empty_results(self):
        assert compute_usage_metrics([]) is None


# ── compute_error_summary ──


class TestComputeErrorSummary:
    def test_no_errors(self):
        results = [_make_result(), _make_result()]
        assert compute_error_summary(results) is None

    def test_with_errors(self):
        error = ErrorInfo(category=ErrorCategory.TARGET, exception_type="TimeoutError", message="timed out")
        results = [
            _make_result(sample_id=0),
            _make_result(sample_id=1, error=error),
            _make_result(sample_id=2, error=error),
        ]
        summary = compute_error_summary(results)
        assert summary is not None
        assert summary.total_errors == 2
        assert summary.by_category == {"target": 2}
        assert summary.by_exception_type == {"TimeoutError": 2}
        assert summary.failed_sample_ids == [1, 2]

    def test_mixed_error_categories(self):
        err_target = ErrorInfo(category=ErrorCategory.TARGET, exception_type="TimeoutError", message="timeout")
        err_grading = ErrorInfo(category=ErrorCategory.GRADING, exception_type="ValueError", message="bad value")
        results = [
            _make_result(sample_id=0, error=err_target),
            _make_result(sample_id=1, error=err_grading),
        ]
        summary = compute_error_summary(results)
        assert summary is not None
        assert summary.total_errors == 2
        assert summary.by_category == {"target": 1, "grading": 1}
        assert summary.by_exception_type == {"TimeoutError": 1, "ValueError": 1}

    def test_empty_results(self):
        assert compute_error_summary([]) is None


# ── compute_timing_metrics ──


class TestComputeTimingMetrics:
    def test_basic(self):
        results = [_make_result(total_time=5.0, target_time=4.0, extraction_time=0.01)]
        timing = compute_timing_metrics(results)
        assert timing is not None
        assert timing.mean_total_seconds == 5.0
        assert timing.mean_target_seconds == 4.0
        assert timing.mean_extraction_seconds == 0.01
        assert timing.p50_total_seconds == 5.0
        assert timing.p95_total_seconds == 5.0

    def test_multiple_results(self):
        results = [
            _make_result(total_time=2.0, target_time=1.5),
            _make_result(total_time=4.0, target_time=3.5),
            _make_result(total_time=6.0, target_time=5.5),
        ]
        timing = compute_timing_metrics(results)
        assert timing is not None
        assert timing.mean_total_seconds == pytest.approx(4.0)
        assert timing.mean_target_seconds == pytest.approx(3.5)
        assert timing.p50_total_seconds == 4.0  # middle value

    def test_p95_with_many_results(self):
        results = [_make_result(total_time=float(i)) for i in range(1, 101)]
        timing = compute_timing_metrics(results)
        assert timing is not None
        # p95 index = min(int(100 * 0.95), 99) = 95, sorted_totals[95] = 96.0
        assert timing.p95_total_seconds == 96.0

    def test_per_grader_time_aggregation(self):
        results = [
            _make_result(per_grader_time={"grader_a": 1.0, "grader_b": 2.0}),
            _make_result(per_grader_time={"grader_a": 3.0, "grader_b": 4.0}),
        ]
        timing = compute_timing_metrics(results)
        assert timing is not None
        assert timing.per_grader_mean_seconds is not None
        assert timing.per_grader_mean_seconds["grader_a"] == pytest.approx(2.0)
        assert timing.per_grader_mean_seconds["grader_b"] == pytest.approx(3.0)

    def test_none_timing_excluded(self):
        results = [_make_result(total_time=None)]
        assert compute_timing_metrics(results) is None

    def test_empty_results(self):
        assert compute_timing_metrics([]) is None

    def test_no_per_grader_time(self):
        results = [_make_result(total_time=5.0, per_grader_time=None)]
        timing = compute_timing_metrics(results)
        assert timing is not None
        assert timing.per_grader_mean_seconds is None


# ── calculate_metrics ──


class TestCalculateMetrics:
    def test_empty_results(self):
        metrics = calculate_metrics([], None, False)
        assert metrics.total == 0
        assert metrics.total_attempted == 0
        assert metrics.avg_score_attempted == 0.0

    def test_single_grader_no_grader_keys(self):
        """When grader_keys is None, uses grade.score directly."""
        results = [
            _make_result(sample_id=0, score=1.0),
            _make_result(sample_id=1, score=0.5),
        ]
        metrics = calculate_metrics(results, None, False)
        assert metrics.total == 2
        assert metrics.total_attempted == 2
        assert metrics.avg_score_attempted == pytest.approx(0.75)
        assert metrics.avg_score_total == pytest.approx(0.75)
        assert "default" in metrics.metrics

    def test_multi_grader(self):
        """When grader_keys is provided, uses grades dict."""
        results = [
            _make_result(
                sample_id=0,
                score=0.8,
                grades={
                    "accuracy": GradeResult(score=0.8, rationale="ok"),
                    "quality": GradeResult(score=0.6, rationale="ok"),
                },
            ),
            _make_result(
                sample_id=1,
                score=1.0,
                grades={
                    "accuracy": GradeResult(score=1.0, rationale="great"),
                    "quality": GradeResult(score=0.9, rationale="great"),
                },
            ),
        ]
        metrics = calculate_metrics(results, ["accuracy", "quality"], False)
        assert metrics.total == 2
        assert metrics.by_metric is not None
        assert metrics.by_metric["accuracy"].avg_score_attempted == pytest.approx(0.9)
        assert metrics.by_metric["quality"].avg_score_attempted == pytest.approx(0.75)
        # first grader is used for overall avg
        assert metrics.avg_score_attempted == pytest.approx(0.9)

    def test_error_results_excluded_from_attempted(self):
        error = ErrorInfo(category=ErrorCategory.TARGET, exception_type="Error", message="fail")
        results = [
            _make_result(sample_id=0, score=1.0),
            _make_result(sample_id=1, score=0.0, error=error),
        ]
        metrics = calculate_metrics(results, None, False)
        assert metrics.total == 2
        assert metrics.total_attempted == 1
        assert metrics.error_summary is not None
        assert metrics.error_summary.total_errors == 1

    def test_per_model_metrics(self):
        results = [
            _make_result(sample_id=0, score=1.0, model_name="gpt-4o"),
            _make_result(sample_id=1, score=0.5, model_name="gpt-4o"),
            _make_result(sample_id=2, score=0.8, model_name="claude"),
        ]
        metrics = calculate_metrics(results, None, has_multi_model=True)
        assert metrics.per_model is not None
        assert len(metrics.per_model) == 2

        gpt_metrics = next(m for m in metrics.per_model if m.model_name == "gpt-4o")
        claude_metrics = next(m for m in metrics.per_model if m.model_name == "claude")

        assert gpt_metrics.total == 2
        assert gpt_metrics.avg_score_attempted == pytest.approx(0.75)
        assert claude_metrics.total == 1
        assert claude_metrics.avg_score_attempted == pytest.approx(0.8)

    def test_no_per_model_when_single_model(self):
        results = [_make_result(model_name="gpt-4o")]
        metrics = calculate_metrics(results, None, has_multi_model=False)
        assert metrics.per_model is None

    def test_usage_and_timing_present(self):
        results = [_make_result(cost=0.01, prompt_tokens=100, total_time=5.0)]
        metrics = calculate_metrics(results, None, False)
        assert metrics.usage_metrics is not None
        assert metrics.timing_metrics is not None
