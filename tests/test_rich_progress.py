from __future__ import annotations

import asyncio

import pytest
from rich.console import Console

from letta_evals.visualization.rich_progress import EvalProgress, SampleProgress, SampleState


class DummyLive:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.stopped = False

    def refresh(self) -> None:
        self.refresh_count += 1

    def stop(self) -> None:
        self.stopped = True


@pytest.mark.asyncio
async def test_update_sample_state_throttles_refreshes() -> None:
    progress = EvalProgress(
        suite_name="demo",
        total_samples=1,
        console=Console(width=120, height=20, force_terminal=False),
        update_freq=10.0,
    )
    progress.main_task_id = progress.main_progress.add_task("Evaluating samples", total=1, completed=0)
    progress.live = DummyLive()  # type: ignore[assignment]

    await progress.update_sample_state(0, SampleState.LOADING_AGENT)
    assert progress.live.refresh_count == 1

    await progress.update_sample_state(0, SampleState.SENDING_MESSAGES, messages_sent=1, total_messages=2)
    assert progress.live.refresh_count == 1

    await asyncio.sleep(0.12)
    assert progress.live.refresh_count == 2

    progress.stop()
    assert progress.live is None


def test_select_display_rows_is_stable_and_recent() -> None:
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

    rows = progress._select_display_rows(limit=5)

    assert [row.sample_id for row in rows] == [0, 3, 5, 2, 7]
