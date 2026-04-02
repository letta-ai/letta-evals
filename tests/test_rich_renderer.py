from __future__ import annotations

import time

from rich.console import Console

from letta_evals.visualization.reducer import ProgressRuntimeState
from letta_evals.visualization.rich_renderer import RichProgressRenderer
from letta_evals.visualization.state import SampleProgress, SampleState


def build_renderer() -> RichProgressRenderer:
    return RichProgressRenderer(
        console=Console(width=120, height=20, force_terminal=False),
        suite_name="demo",
        target_kind="agent",
        grader_kind="tool",
        rubric_model=None,
        max_concurrent=8,
    )


def build_runtime_state() -> ProgressRuntimeState:
    return ProgressRuntimeState(
        samples={
            (0, None): SampleProgress(
                sample_id=0,
                state=SampleState.GRADING,
                last_update_ts=30.0,
                active_sort_sequence=2,
            ),
            (3, None): SampleProgress(
                sample_id=3,
                state=SampleState.SENDING_MESSAGES,
                last_update_ts=10.0,
                active_sort_sequence=1,
            ),
            (5, None): SampleProgress(
                sample_id=5,
                state=SampleState.COMPLETED,
                last_update_ts=80.0,
                completion_sequence=3,
            ),
            (2, None): SampleProgress(
                sample_id=2,
                state=SampleState.ERROR,
                last_update_ts=70.0,
                completion_sequence=2,
            ),
            (7, None): SampleProgress(
                sample_id=7,
                state=SampleState.COMPLETED,
                last_update_ts=999.0,
                completion_sequence=1,
            ),
            (6, None): SampleProgress(sample_id=6, state=SampleState.QUEUED, last_update_ts=5.0),
        }
    )


def test_select_active_rows_excludes_completed_and_queued() -> None:
    renderer = build_renderer()

    rows = renderer.select_active_rows(build_runtime_state(), limit=5)

    assert [row.sample_id for row in rows] == [3, 0]


def test_select_completed_rows_is_recent_first() -> None:
    renderer = build_renderer()

    rows = renderer.select_completed_rows(build_runtime_state(), limit=5)

    assert [row.sample_id for row in rows] == [5, 2, 7]


def test_create_samples_table_pads_rows_to_fixed_limit() -> None:
    renderer = build_renderer()
    table = renderer._create_samples_table(  # noqa: SLF001
        [SampleProgress(sample_id=0, state=SampleState.LOADING_AGENT)],
        "Active Samples",
        row_limit=3,
    )

    assert table.title == "Active Samples"
    assert len(table.rows) == 3
    assert table.columns[1].no_wrap is True
    assert table.columns[-1].no_wrap is True


def test_visible_snapshot_ignores_offscreen_row_updates() -> None:
    renderer = RichProgressRenderer(
        console=Console(width=120, height=12, force_terminal=False),
        suite_name="demo",
        target_kind="agent",
        grader_kind="tool",
        rubric_model=None,
        max_concurrent=8,
    )
    state = ProgressRuntimeState(
        samples={
            (0, None): SampleProgress(sample_id=0, state=SampleState.LOADING_AGENT, active_sort_sequence=1),
            (1, None): SampleProgress(sample_id=1, state=SampleState.LOADING_AGENT, active_sort_sequence=2),
            (2, None): SampleProgress(sample_id=2, state=SampleState.LOADING_AGENT, active_sort_sequence=3),
        }
    )

    snapshot_before = renderer.build_visible_snapshot(state)
    state.samples[(2, None)].agent_id = "agent-offscreen"
    state.samples[(2, None)].messages_sent = 5
    state.samples[(2, None)].total_messages = 10
    snapshot_after = renderer.build_visible_snapshot(state)

    assert snapshot_before == snapshot_after


def test_visible_snapshot_does_not_drift_with_elapsed_time_only() -> None:
    renderer = build_renderer()
    started = time.time() - 5
    state = ProgressRuntimeState(
        samples={
            (0, None): SampleProgress(
                sample_id=0,
                state=SampleState.GRADING,
                start_time=started,
                active_sort_sequence=1,
            )
        }
    )

    snapshot_before = renderer.build_visible_snapshot(state)
    time.sleep(0.02)
    snapshot_after = renderer.build_visible_snapshot(state)

    assert snapshot_before == snapshot_after
