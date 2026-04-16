from __future__ import annotations

from letta_evals.visualization.reducer import ProgressRuntimeState, ProgressStateReducer
from letta_evals.visualization.state import ProgressEvent, SampleState


def test_apply_completed_update_tracks_counts_and_metric_totals() -> None:
    reducer = ProgressStateReducer(ProgressRuntimeState())

    result = reducer.apply_event(
        ProgressEvent(
            kind="update_sample_state",
            payload={
                "sample_id": 0,
                "state": SampleState.COMPLETED,
                "score": 0.75,
                "target_cost": 0.0123,
                "metric_scores": {"accuracy": 1.0, "fluency": 0.5},
            },
        )
    )

    assert reducer.state.completed_count == 1
    assert reducer.state.error_count == 0
    assert reducer.state.total_target_cost == 0.0123
    assert reducer.state.metric_totals == {"accuracy": 1.0, "fluency": 0.5}
    assert reducer.state.metric_counts == {"accuracy": 1, "fluency": 1}
    assert result.progress_completed == 1


def test_apply_error_update_tracks_progress_without_incrementing_completed() -> None:
    reducer = ProgressStateReducer(ProgressRuntimeState())

    result = reducer.apply_event(
        ProgressEvent(
            kind="update_sample_state",
            payload={"sample_id": 1, "state": SampleState.ERROR, "error": "boom", "target_cost": 0.0456},
        )
    )

    assert reducer.state.completed_count == 0
    assert reducer.state.error_count == 1
    assert reducer.state.total_target_cost == 0.0456
    assert result.progress_completed == 1


def test_duplicate_terminal_updates_do_not_double_count_target_cost() -> None:
    reducer = ProgressStateReducer(ProgressRuntimeState())

    reducer.apply_event(
        ProgressEvent(
            kind="update_sample_state",
            payload={"sample_id": 9, "state": SampleState.COMPLETED, "target_cost": 0.02},
        )
    )
    reducer.apply_event(
        ProgressEvent(
            kind="update_sample_state",
            payload={"sample_id": 9, "state": SampleState.COMPLETED, "target_cost": 0.03},
        )
    )

    assert reducer.state.completed_count == 1
    assert reducer.state.total_target_cost == 0.02


def test_ensure_sample_migrates_placeholder_when_model_name_arrives() -> None:
    reducer = ProgressStateReducer(ProgressRuntimeState())
    reducer.ensure_sample(2, agent_id="agent-1")

    sample = reducer.ensure_sample(2, agent_id="agent-1", model_name="gpt-5")

    assert (2, None) not in reducer.state.samples
    assert reducer.state.samples[(2, "gpt-5")] is sample
    assert sample.model_name == "gpt-5"


def test_get_from_cache_checks_model_specific_and_placeholder_entries() -> None:
    reducer = ProgressStateReducer(ProgressRuntimeState())
    reducer.ensure_sample(3, model_name=None).from_cache = True

    assert reducer.get_from_cache(3, None) is True
    assert reducer.get_from_cache(3, "gpt-5") is True
    assert reducer.get_from_cache(999, "gpt-5") is False


def test_record_turn_grade_initializes_and_updates_turn_scores() -> None:
    reducer = ProgressStateReducer(ProgressRuntimeState())

    sample = reducer.record_turn_grade(
        4,
        turn_num=1,
        total_turns=3,
        turn_score=0.8,
        grader_key="safety",
        agent_id="agent-4",
        model_name="gpt-5",
    )

    assert sample.agent_id == "agent-4"
    assert sample.model_name == "gpt-5"
    assert sample.turn_scores == {"safety": [None, 0.8, None]}
