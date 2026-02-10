from __future__ import annotations

import io

import pytest
from rich.console import Console

from letta_evals.visualization.rich_progress import EvalProgress, SampleState


class DummyLive:
    def __init__(self) -> None:
        self.calls = 0

    def update(self, _renderable, *, refresh: bool = False) -> None:
        self.calls += 1


@pytest.mark.asyncio
async def test_update_sample_state_noop_is_tracked() -> None:
    progress = EvalProgress("suite", 10, console=Console(file=io.StringIO()), show_samples=False)

    await progress.update_sample_state(0, SampleState.LOADING_AGENT, agent_id="a", model_name="m")
    await progress.update_sample_state(0, SampleState.LOADING_AGENT, agent_id="a", model_name="m")

    snapshot = progress.get_perf_snapshot()
    assert snapshot["noop_state_updates"] == 1


@pytest.mark.asyncio
async def test_callbacks_mark_dirty_without_immediate_render() -> None:
    progress = EvalProgress("suite", 10, console=Console(file=io.StringIO()), show_samples=False)
    progress.live = DummyLive()

    await progress.message_sending(sample_id=0, message_num=1, total_messages=3, agent_id="a", model_name="m")

    assert progress.live.calls == 0
    assert progress._dirty is True

    await progress._render_if_dirty()
    assert progress.live.calls == 1


@pytest.mark.asyncio
async def test_render_if_dirty_renders_once_per_dirty_cycle() -> None:
    progress = EvalProgress("suite", 10, console=Console(file=io.StringIO()), show_samples=False)
    progress.live = DummyLive()
    progress._dirty = True

    await progress._render_if_dirty()
    await progress._render_if_dirty()

    assert progress.live.calls == 1
    assert progress.get_perf_snapshot()["renders_total"] == 1


@pytest.mark.asyncio
async def test_sample_completed_counts_and_metrics_once() -> None:
    progress = EvalProgress("suite", 10, console=Console(file=io.StringIO()), show_samples=False)

    await progress.sample_completed(
        sample_id=1,
        agent_id="a",
        score=0.5,
        model_name="m",
        metric_scores={"quality": 0.5},
        metric_rationales={"quality": "ok"},
    )
    await progress.sample_completed(
        sample_id=1,
        agent_id="a",
        score=0.5,
        model_name="m",
        metric_scores={"quality": 0.5},
        metric_rationales={"quality": "ok"},
    )

    assert progress.completed_count == 1
    assert progress.metric_totals["quality"] == 0.5
    assert progress.metric_counts["quality"] == 1
