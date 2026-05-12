"""Unit tests for letta_evals.metrics module (post data-model overhaul).

The new API is built around four primitives:

  - ``aggregate_usage``    : sum token/cost fields across samples
  - ``aggregate_timing``   : produce TimingStats from per-sample Timing
  - ``aggregate_errors``   : produce ErrorSummary (or None)
  - ``summarize_model``    : single-run ModelSummary
  - ``summarize_run``      : single PerRunSummary
  - ``summarize_runs``     : multi-run ModelSummary (with std + per-run list)

All scores live on a 0-1 scale.
"""

import math

import pytest

from letta_evals.metrics import (
    aggregate_errors,
    aggregate_timing,
    aggregate_usage,
    summarize_model,
    summarize_run,
    summarize_runs,
)
from letta_evals.models import (
    Error,
    GradeResult,
    SampleResult,
    SimpleGateSpec,
    Timing,
    Usage,
    WeightedAverageGateSpec,
)
from letta_evals.types import Aggregation, ErrorCategory, GateKind, MetricOp


_DEFAULT_GRADER = "default"


def _make_result(
    sample_id: int = 0,
    score: float = 1.0,
    *,
    grades: dict | None = None,
    usage: Usage | None | object = object(),
    total_time: float | None = 5.0,
    target_time: float | None = 4.5,
    extraction_time: float | None = 0.001,
    per_grader_time: dict | None | object = object(),
    error: Error | None = None,
) -> SampleResult:
    if grades is None:
        grades = {_DEFAULT_GRADER: GradeResult(score=score, rationale="test")}

    if usage is object:  # sentinel = use default Usage
        usage = Usage()
    if isinstance(usage, object) and not isinstance(usage, (Usage, type(None))):
        # the sentinel default
        usage = Usage(prompt_tokens=100, completion_tokens=50, cost=0.01)

    if per_grader_time is object:
        per_grader_time = {"grader_a": 0.01}

    timing = None
    if total_time is not None:
        timing = Timing(
            total=total_time,
            target=target_time if target_time is not None else 0.0,
            extraction=extraction_time,
            per_grader=per_grader_time if isinstance(per_grader_time, dict) else None,
        )

    return SampleResult(
        sample_id=sample_id,
        trajectory=[[]],
        submissions={k: "x" for k in grades},
        grades=grades,
        usage=usage if isinstance(usage, Usage) else None,
        timing=timing or Timing(total=0.0, target=0.0),
        error=error,
    )


# ── aggregate_usage ──


class TestAggregateUsage:
    def test_sums_tokens_and_cost(self):
        results = [
            _make_result(usage=Usage(prompt_tokens=100, completion_tokens=50, cost=0.01)),
            _make_result(usage=Usage(prompt_tokens=200, completion_tokens=100, cost=0.02)),
        ]
        u = aggregate_usage(results)
        assert u.prompt_tokens == 300
        assert u.completion_tokens == 150
        assert u.cost == pytest.approx(0.03)

    def test_no_cost_means_cost_is_none(self):
        results = [_make_result(usage=Usage(prompt_tokens=10, completion_tokens=5, cost=None))]
        u = aggregate_usage(results)
        assert u.prompt_tokens == 10
        assert u.cost is None

    def test_cached_and_reasoning_tokens(self):
        results = [
            _make_result(usage=Usage(
                prompt_tokens=100, completion_tokens=50,
                cached_input_tokens=30, cache_write_tokens=20, reasoning_tokens=10,
            )),
            _make_result(usage=Usage(
                prompt_tokens=200, completion_tokens=100,
                cached_input_tokens=60, cache_write_tokens=40, reasoning_tokens=25,
            )),
        ]
        u = aggregate_usage(results)
        assert u.cached_input_tokens == 90
        assert u.cache_write_tokens == 60
        assert u.reasoning_tokens == 35

    def test_empty_returns_zeroed_usage(self):
        u = aggregate_usage([])
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.cost is None


# ── aggregate_errors ──


class TestAggregateErrors:
    def test_no_errors_returns_none(self):
        assert aggregate_errors([_make_result(), _make_result()]) is None

    def test_with_errors(self):
        err = Error(category=ErrorCategory.TARGET, exception_type="TimeoutError", message="timed out")
        results = [
            _make_result(sample_id=0),
            _make_result(sample_id=1, error=err),
            _make_result(sample_id=2, error=err),
        ]
        s = aggregate_errors(results)
        assert s is not None
        assert s.total_errors == 2
        assert s.by_category == {"target": 2}
        assert s.by_exception_type == {"TimeoutError": 2}

    def test_mixed_categories(self):
        e1 = Error(category=ErrorCategory.TARGET, exception_type="TimeoutError", message="x")
        e2 = Error(category=ErrorCategory.GRADING, exception_type="ValueError", message="y")
        s = aggregate_errors([_make_result(error=e1), _make_result(error=e2)])
        assert s is not None
        assert s.by_category == {"target": 1, "grading": 1}
        assert s.by_exception_type == {"TimeoutError": 1, "ValueError": 1}


# ── aggregate_timing ──


class TestAggregateTiming:
    def test_basic(self):
        t = aggregate_timing([_make_result(total_time=5.0, target_time=4.0, extraction_time=0.01)])
        assert t is not None
        assert t.mean_total == 5.0
        assert t.mean_target == 4.0
        assert t.mean_extraction == 0.01

    def test_mean_p50_p95(self):
        results = [_make_result(total_time=float(i)) for i in range(1, 101)]
        t = aggregate_timing(results)
        assert t is not None
        # p95 index = min(int(100 * 0.95), 99) = 95, sorted_totals[95] = 96
        assert t.p95_total == 96.0

    def test_per_grader_mean(self):
        results = [
            _make_result(per_grader_time={"grader_a": 1.0, "grader_b": 2.0}),
            _make_result(per_grader_time={"grader_a": 3.0, "grader_b": 4.0}),
        ]
        t = aggregate_timing(results)
        assert t is not None
        assert t.per_grader_mean is not None
        assert t.per_grader_mean["grader_a"] == pytest.approx(2.0)
        assert t.per_grader_mean["grader_b"] == pytest.approx(3.0)

    def test_empty_returns_none(self):
        assert aggregate_timing([]) is None


# ── summarize_model (single run) ──


class TestSummarizeModel:
    def test_simple_gate_score(self):
        gate = SimpleGateSpec(
            kind=GateKind.SIMPLE,
            metric_key="accuracy",
            aggregation=Aggregation.AVG_SCORE,
            op=MetricOp.GTE,
            value=0.5,
        )
        results = [
            _make_result(sample_id=0, grades={
                "accuracy": GradeResult(score=0.8, rationale="ok"),
                "quality": GradeResult(score=0.6, rationale="ok"),
            }),
            _make_result(sample_id=1, grades={
                "accuracy": GradeResult(score=1.0, rationale="great"),
                "quality": GradeResult(score=0.9, rationale="great"),
            }),
        ]
        s = summarize_model(model="gpt-4o", results=results, grader_keys=["accuracy", "quality"], gate=gate)
        assert s.model == "gpt-4o"
        assert s.n_total == 2
        assert s.n_attempted == 2
        assert s.score == pytest.approx(0.9)  # accuracy mean
        assert s.per_metric["accuracy"] == pytest.approx(0.9)
        assert s.per_metric["quality"] == pytest.approx(0.75)

    def test_weighted_average_gate_score(self):
        gate = WeightedAverageGateSpec(
            kind=GateKind.WEIGHTED_AVERAGE,
            weights={"accuracy": 0.7, "quality": 0.3},
            aggregation=Aggregation.AVG_SCORE,
            op=MetricOp.GTE,
            value=0.5,
        )
        results = [
            _make_result(grades={
                "accuracy": GradeResult(score=0.8),
                "quality": GradeResult(score=0.4),
            }),
        ]
        s = summarize_model(model="m", results=results, grader_keys=["accuracy", "quality"], gate=gate)
        assert s.score == pytest.approx(0.7 * 0.8 + 0.3 * 0.4)

    def test_no_gate_uses_mean(self):
        results = [
            _make_result(grades={
                "a": GradeResult(score=1.0),
                "b": GradeResult(score=0.5),
            }),
        ]
        s = summarize_model(model="m", results=results, grader_keys=["a", "b"], gate=None)
        assert s.score == pytest.approx(0.75)

    def test_errors_excluded_from_attempted(self):
        err = Error(category=ErrorCategory.TARGET, exception_type="X", message="boom")
        gate = SimpleGateSpec(
            kind=GateKind.SIMPLE, metric_key="default", aggregation=Aggregation.AVG_SCORE,
            op=MetricOp.GTE, value=0.5,
        )
        results = [
            _make_result(sample_id=0, score=1.0),
            _make_result(sample_id=1, score=0.0, error=err),
        ]
        s = summarize_model(model="m", results=results, grader_keys=[_DEFAULT_GRADER], gate=gate)
        assert s.n_total == 2
        assert s.n_attempted == 1
        assert s.errors is not None
        assert s.errors.total_errors == 1
        # Score over attempted only
        assert s.score == pytest.approx(1.0)

    def test_empty_results(self):
        s = summarize_model(model="m", results=[], grader_keys=["default"], gate=None)
        assert s.n_total == 0
        assert s.n_attempted == 0
        assert s.score == 0.0
        # Timing falls back to zeroed TimingStats
        assert s.timing.mean_total == 0.0


# ── summarize_run / summarize_runs (multi-run) ──


class TestSummarizeRuns:
    def _gate(self) -> SimpleGateSpec:
        return SimpleGateSpec(
            kind=GateKind.SIMPLE, metric_key="accuracy", aggregation=Aggregation.AVG_SCORE,
            op=MetricOp.GTE, value=0.5,
        )

    def test_summarize_run_returns_per_run_summary(self):
        results = [_make_result(grades={"accuracy": GradeResult(score=1.0)})]
        ps = summarize_run(run=1, results=results, grader_keys=["accuracy"], gate=self._gate())
        assert ps.run == 1
        assert ps.score == pytest.approx(1.0)
        assert ps.per_metric["accuracy"] == pytest.approx(1.0)
        assert ps.n_errors == 0

    def test_summarize_runs_score_std(self):
        gate = self._gate()
        run1 = [_make_result(grades={"accuracy": GradeResult(score=1.0)})]
        run2 = [_make_result(grades={"accuracy": GradeResult(score=0.0)})]
        s = summarize_runs(model="m", per_run_results=[run1, run2], grader_keys=["accuracy"], gate=gate)
        assert s.score == pytest.approx(0.5)
        assert s.score_std == pytest.approx(0.7071067811865476, rel=1e-3)  # stdev([1, 0])
        assert s.runs is not None and len(s.runs) == 2
        assert s.runs[0].score == 1.0 and s.runs[1].score == 0.0

    def test_summarize_runs_per_metric_std(self):
        gate = self._gate()
        run1 = [_make_result(grades={"accuracy": GradeResult(score=0.8), "quality": GradeResult(score=0.5)})]
        run2 = [_make_result(grades={"accuracy": GradeResult(score=0.4), "quality": GradeResult(score=0.5)})]
        s = summarize_runs(model="m", per_run_results=[run1, run2], grader_keys=["accuracy", "quality"], gate=gate)
        assert s.per_metric["accuracy"] == pytest.approx(0.6)
        assert s.per_metric["quality"] == pytest.approx(0.5)
        assert s.per_metric_std is not None
        assert s.per_metric_std["accuracy"] > 0
        assert s.per_metric_std["quality"] == 0.0

    def test_summarize_runs_empty_raises(self):
        with pytest.raises(ValueError):
            summarize_runs(model="m", per_run_results=[], grader_keys=["x"], gate=None)
