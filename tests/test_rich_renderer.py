from __future__ import annotations

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
            (0, None): SampleProgress(sample_id=0, state=SampleState.GRADING, last_update_ts=30.0),
            (3, None): SampleProgress(sample_id=3, state=SampleState.SENDING_MESSAGES, last_update_ts=10.0),
            (5, None): SampleProgress(sample_id=5, state=SampleState.COMPLETED, last_update_ts=80.0),
            (2, None): SampleProgress(sample_id=2, state=SampleState.ERROR, last_update_ts=70.0),
            (7, None): SampleProgress(sample_id=7, state=SampleState.COMPLETED, last_update_ts=60.0),
            (6, None): SampleProgress(sample_id=6, state=SampleState.QUEUED, last_update_ts=5.0),
        }
    )


def test_select_active_rows_excludes_completed_and_queued() -> None:
    renderer = build_renderer()

    rows = renderer.select_active_rows(build_runtime_state(), limit=5)

    assert [row.sample_id for row in rows] == [0, 3]


def test_select_completed_rows_is_recent_first() -> None:
    renderer = build_renderer()

    rows = renderer.select_completed_rows(build_runtime_state(), limit=5)

    assert [row.sample_id for row in rows] == [5, 2, 7]
