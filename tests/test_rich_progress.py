from __future__ import annotations

import asyncio

import pytest
from rich.console import Console

from letta_evals.visualization.rich_progress import EvalProgress
from letta_evals.visualization.state import SampleProgress, SampleState


class DummyLive:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.stopped = False

    def refresh(self) -> None:
        self.refresh_count += 1

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_event_loop_batches_burst_updates_into_single_refresh() -> None:
    progress = EvalProgress(
        suite_name="demo",
        total_samples=1,
        console=Console(width=120, height=20, force_terminal=False),
        update_freq=5.0,
    )
    progress.main_task_id = progress.main_progress.add_task("Evaluating samples", total=1, completed=0)
    progress.live = DummyLive()  # type: ignore[assignment]
    progress._start_background_tasks()

    await progress.update_sample_state(0, SampleState.LOADING_AGENT)
    await progress.update_sample_state(0, SampleState.SENDING_MESSAGES, messages_sent=1, total_messages=2)
    await progress.update_sample_state(0, SampleState.COMPLETED, score=1.0)
    assert progress.live.refresh_count == 0

    await asyncio.sleep(0.25)
    assert progress.live.refresh_count == 1
    stats = progress.get_stats_snapshot()
    assert stats.events_emitted == 3
    assert stats.events_processed == 3
    assert stats.refreshes == 1
    assert stats.avg_events_per_refresh == 3.0

    progress.stop()
    assert progress.live is None


@pytest.mark.asyncio
async def test_stop_flushes_dirty_state_immediately() -> None:
    progress = EvalProgress(
        suite_name="demo",
        total_samples=1,
        console=Console(width=120, height=20, force_terminal=False),
        update_freq=5.0,
    )
    progress.main_task_id = progress.main_progress.add_task("Evaluating samples", total=1, completed=0)
    live = DummyLive()
    progress.live = live  # type: ignore[assignment]
    progress._start_background_tasks()

    await progress.update_sample_state(0, SampleState.LOADING_AGENT)
    assert live.refresh_count == 0

    progress.stop()

    assert live.refresh_count == 1
    assert live.stopped
    assert progress.live is None
    stats = progress.get_stats_snapshot()
    assert stats.refreshes == 1
    assert stats.refreshes_by_reason == {"stop": 1}


@pytest.mark.asyncio
async def test_stats_capture_queue_pressure_under_concurrency() -> None:
    progress = EvalProgress(
        suite_name="demo",
        total_samples=15,
        console=Console(width=120, height=20, force_terminal=False),
        update_freq=20.0,
    )
    progress.main_task_id = progress.main_progress.add_task("Evaluating samples", total=15, completed=0)
    progress.live = DummyLive()  # type: ignore[assignment]
    progress._start_background_tasks()

    await asyncio.gather(
        *[progress.update_sample_state(i, SampleState.LOADING_AGENT) for i in range(15)],
    )

    await asyncio.sleep(0.1)
    stats = progress.get_stats_snapshot()

    assert stats.events_emitted == 15
    assert stats.events_processed == 15
    assert stats.max_queue_depth >= 1
    assert stats.refreshes < stats.events_processed

    progress.stop()


def test_select_active_rows_excludes_completed_and_queued() -> None:
    progress = EvalProgress(
        suite_name="demo",
        total_samples=8,
        console=Console(width=120, height=20, force_terminal=False),
    )
    progress.samples = {
        (0, None): SampleProgress(sample_id=0, state=SampleState.GRADING, last_update_ts=30.0),
        (3, None): SampleProgress(sample_id=3, state=SampleState.SENDING_MESSAGES, last_update_ts=10.0),
        (5, None): SampleProgress(sample_id=5, state=SampleState.COMPLETED, last_update_ts=80.0),
        (2, None): SampleProgress(sample_id=2, state=SampleState.ERROR, last_update_ts=70.0),
        (7, None): SampleProgress(sample_id=7, state=SampleState.COMPLETED, last_update_ts=60.0),
        (6, None): SampleProgress(sample_id=6, state=SampleState.QUEUED, last_update_ts=5.0),
    }

    rows = progress._select_active_rows(limit=5)

    assert [row.sample_id for row in rows] == [0, 3]


def test_select_completed_rows_is_recent_first() -> None:
    progress = EvalProgress(
        suite_name="demo",
        total_samples=8,
        console=Console(width=120, height=20, force_terminal=False),
    )
    progress.samples = {
        (0, None): SampleProgress(sample_id=0, state=SampleState.GRADING, last_update_ts=30.0),
        (3, None): SampleProgress(sample_id=3, state=SampleState.SENDING_MESSAGES, last_update_ts=10.0),
        (5, None): SampleProgress(sample_id=5, state=SampleState.COMPLETED, last_update_ts=80.0),
        (2, None): SampleProgress(sample_id=2, state=SampleState.ERROR, last_update_ts=70.0),
        (7, None): SampleProgress(sample_id=7, state=SampleState.COMPLETED, last_update_ts=60.0),
        (6, None): SampleProgress(sample_id=6, state=SampleState.QUEUED, last_update_ts=5.0),
    }

    rows = progress._select_completed_rows(limit=5)

    assert [row.sample_id for row in rows] == [5, 2, 7]
