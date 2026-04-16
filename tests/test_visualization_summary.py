from __future__ import annotations

from types import SimpleNamespace

from letta_evals.visualization.summary import (
    build_rich_sample_results_table,
    build_simple_sample_results_table,
    format_gate_description,
    get_displayed_sample_results,
)


def _make_result():
    return SimpleNamespace(
        config={
            "gate": {
                "kind": "simple",
                "metric_key": "accuracy",
                "aggregation": "avg_score",
                "op": "gte",
                "value": 0.5,
            },
            "graders": {
                "accuracy": {"display_name": "Accuracy"},
                "quality": {"display_name": "Quality"},
            },
        },
        results=[
            SimpleNamespace(
                sample=SimpleNamespace(id=1),
                agent_id="agent-b",
                model_name="model-z",
                grades={
                    "accuracy": {"score": 0.25, "rationale": "missed details"},
                    "quality": {"score": 0.75, "rationale": "solid"},
                },
            ),
            SimpleNamespace(
                sample=SimpleNamespace(id=0),
                agent_id="agent-a",
                model_name="model-a",
                grades={
                    "accuracy": SimpleNamespace(score=1.0, rationale="perfect"),
                    "quality": SimpleNamespace(score=0.5, rationale="fine"),
                },
            ),
        ],
    )


def test_format_gate_description_for_simple_progress() -> None:
    result = _make_result()

    description = format_gate_description(result.config, fixed_decimal_value=True)

    assert description == "accuracy avg_score ≥ 0.50"


def test_format_gate_description_for_rich_progress() -> None:
    result = _make_result()

    description = format_gate_description(
        result.config,
        prefer_display_label=True,
        quote_metric_label=True,
        default_metric_label="metric",
    )

    assert description == "'Accuracy' avg_score ≥ 0.5"


def test_get_displayed_sample_results_sorts_by_model_then_sample() -> None:
    result = _make_result()

    total, displayed = get_displayed_sample_results(result)

    assert total == 2
    assert [(sample_result.model_name, sample_result.sample.id) for sample_result in displayed] == [
        ("model-a", 0),
        ("model-z", 1),
    ]


def test_build_simple_sample_results_table_contains_scores_only() -> None:
    result = _make_result()
    _, displayed_results = get_displayed_sample_results(result)

    table = build_simple_sample_results_table(result.config, displayed_results)

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
    _, displayed_results = get_displayed_sample_results(result)

    table = build_rich_sample_results_table(result.config, displayed_results)

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
