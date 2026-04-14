from __future__ import annotations

from rich.console import Console, Group
from rich.progress import Progress

from letta_evals.types import GraderKind
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


def test_render_splits_active_and_completed_panels_evenly() -> None:
    renderer = build_renderer()

    layout = renderer.render(build_runtime_state(), Progress())
    detail_panel = layout["details"]

    assert detail_panel.children[0].ratio == 1
    assert detail_panel.children[1].ratio == 1
    active_size, completed_size = renderer._detail_layout_budget(build_runtime_state())
    assert active_size == completed_size


def test_model_judge_renderer_moves_rubric_model_into_header_and_uses_fixed_columns() -> None:
    console = Console(width=160, height=20, force_terminal=False, record=True)
    renderer = RichProgressRenderer(
        console=console,
        suite_name="demo",
        target_kind="agent",
        grader_kind=GraderKind.MODEL_JUDGE.value,
        rubric_model="judge-model-v1",
        max_concurrent=8,
        metric_labels={"accuracy": "Accuracy"},
    )

    sample = SampleProgress(
        sample_id=0,
        state=SampleState.COMPLETED,
        agent_id="agent-with-a-very-long-id",
        model_name="target-model-with-a-long-name",
        metric_scores={"accuracy": 1.0},
        metric_rationales={"accuracy": "This is a deliberately long rationale that should be truncated by Rich."},
        rationale="unused",
    )
    table = renderer._create_samples_table([sample], "Recent Completed", "None")

    assert [column.header for column in table.columns] == [
        "#",
        "Agent ID",
        "Model",
        "Status",
        "Accuracy Score",
        "Accuracy Rationale",
        "Time",
        "Details",
    ]
    assert table.expand
    assert [column.width for column in table.columns] == [3, 14, 14, 11, 10, 20, 6, 14]
    assert [column.ratio for column in table.columns] == [None, 2, 2, None, None, 3, None, 2]
    assert [column.max_width for column in table.columns] == [3, None, None, 11, 10, None, 6, None]
    assert all(column.no_wrap for column in table.columns)
    assert all(column.overflow == "ellipsis" for column in table.columns)

    header_panel = renderer._create_header_panel()
    assert isinstance(header_panel.renderable, Group)
    console.print(header_panel)
    assert "Rubric: judge-model-v1" in console.export_text()


def test_progress_panel_renders_live_target_cost() -> None:
    console = Console(width=120, height=20, force_terminal=False, record=True)
    renderer = RichProgressRenderer(
        console=console,
        suite_name="demo",
        target_kind="agent",
        grader_kind="tool",
        rubric_model=None,
        max_concurrent=8,
    )
    runtime_state = ProgressRuntimeState(total_target_cost=0.0342, completed_count=2, error_count=1)
    progress = Progress()

    console.print(renderer._create_progress_with_metrics(runtime_state, progress))

    assert "Target cost: $0.0342" in console.export_text()
