from __future__ import annotations

from types import SimpleNamespace

from letta_evals.models import (
    GradeResult,
    LettaAgentTargetSpec,
    ModelRun,
    ModelSummary,
    SampleResult,
    SimpleGateSpec,
    SuiteSpec,
    Timing,
    TimingStats,
    ToolGraderSpec,
    Usage,
)
from letta_evals.types import (
    Aggregation,
    GateKind,
    GraderKind,
    MetricOp,
    TargetKind,
)
from letta_evals.visualization.summary import (
    build_rich_sample_results_table,
    build_simple_sample_results_table,
    format_gate_description,
    get_displayed_sample_results,
)


def _make_suite() -> SuiteSpec:
    return SuiteSpec(
        name="test-suite",
        dataset="ignored",
        target=LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_id="agent-fake-1"),
        graders={
            "accuracy": ToolGraderSpec(function="exact_match", display_name="Accuracy"),
            "quality": ToolGraderSpec(function="exact_match", display_name="Quality"),
        },
        gate=SimpleGateSpec(
            kind=GateKind.SIMPLE,
            metric_key="accuracy",
            aggregation=Aggregation.AVG_SCORE,
            op=MetricOp.GTE,
            value=0.5,
        ),
    )


def _sample_result(sample_id: int, agent_id: str, accuracy: float, acc_rat: str, quality: float, q_rat: str) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        agent_id=agent_id,
        trajectory=[[]],
        submissions={"accuracy": "x", "quality": "x"},
        grades={
            "accuracy": GradeResult(score=accuracy, rationale=acc_rat),
            "quality": GradeResult(score=quality, rationale=q_rat),
        },
        timing=Timing(total=0.0, target=0.0),
    )


def _model_summary(model_id: str, score: float, per_metric: dict) -> ModelSummary:
    return ModelSummary(
        model=model_id,
        n_total=1,
        n_attempted=1,
        score=score,
        per_metric=per_metric,
        usage=Usage(),
        timing=TimingStats(mean_total=0.0, mean_target=0.0, p50_total=0.0, p95_total=0.0),
    )


def _make_result():
    """Build a fake RunnerResult-like SimpleNamespace with the new shape."""
    suite_spec = _make_suite()
    model_a = ModelRun(
        model="model-a",
        results=[_sample_result(0, "agent-a", accuracy=1.0, acc_rat="perfect", quality=0.5, q_rat="fine")],
        summary=_model_summary("model-a", score=1.0, per_metric={"accuracy": 1.0, "quality": 0.5}),
    )
    model_z = ModelRun(
        model="model-z",
        results=[_sample_result(1, "agent-b", accuracy=0.25, acc_rat="missed details", quality=0.75, q_rat="solid")],
        summary=_model_summary("model-z", score=0.25, per_metric={"accuracy": 0.25, "quality": 0.75}),
    )
    return SimpleNamespace(
        suite_spec=suite_spec,
        runs={"model-a": model_a, "model-z": model_z},
    )


def test_format_gate_description_for_simple_progress() -> None:
    result = _make_result()

    description = format_gate_description(result.suite_spec, fixed_decimal_value=True)

    assert description == "accuracy avg_score ≥ 0.50"


def test_format_gate_description_for_rich_progress() -> None:
    result = _make_result()

    description = format_gate_description(
        result.suite_spec,
        prefer_display_label=True,
        quote_metric_label=True,
        default_metric_label="metric",
    )

    assert description == "'Accuracy' avg_score ≥ 0.5"


def test_get_displayed_sample_results_sorts_by_model_then_sample() -> None:
    result = _make_result()

    total, displayed = get_displayed_sample_results(result)

    assert total == 2
    assert [(model_id, sr.sample_id) for model_id, sr in displayed] == [
        ("model-a", 0),
        ("model-z", 1),
    ]


def test_build_simple_sample_results_table_contains_scores_only() -> None:
    result = _make_result()
    _, displayed_rows = get_displayed_sample_results(result)

    table = build_simple_sample_results_table(result.suite_spec, displayed_rows)

    assert [column.header for column in table.columns] == [
        "Sample",
        "Agent ID",
        "Model",
        "Accuracy score",
        "Quality score",
    ]
    assert table.columns[0]._cells == ["Sample 0", "Sample 1"]
    assert table.columns[3]._cells == ["1.00", "0.25"]
    assert table.columns[4]._cells == ["0.50", "0.75"]


def test_build_rich_sample_results_table_contains_rationales() -> None:
    result = _make_result()
    _, displayed_rows = get_displayed_sample_results(result)

    table = build_rich_sample_results_table(result.suite_spec, displayed_rows)

    assert [column.header for column in table.columns] == [
        "Sample",
        "Agent ID",
        "Model",
        "Accuracy score",
        "Accuracy rationale",
        "Quality score",
        "Quality rationale",
    ]
    assert table.columns[4]._cells == ["perfect", "missed details"]
    assert table.columns[6]._cells == ["fine", "solid"]
