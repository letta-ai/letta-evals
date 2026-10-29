"""Runner on backend: local — the trace comes from the target, nothing is read
from or deleted on a server, and cleanup removes the sample's local store."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from letta_client.types.agents import AssistantMessage

from letta_evals.models import RewardOutput, Sample, TargetTrace
from letta_evals.targets.errors import TargetError
from tests.test_runner_token_data import _FakeGrader, _make_runner

MESSAGE = AssistantMessage(
    id="assistant-1", date=datetime.now(timezone.utc), content="done", message_type="assistant_message"
)
USAGE = [{"message_type": "usage_statistics", "prompt_tokens": 5, "completion_tokens": 2}]


def _local_runner(monkeypatch, target: MagicMock, *, cleanup: bool = False):
    for name in ("fetch_trajectory", "fetch_token_data"):
        monkeypatch.setattr(f"letta_evals.runner.{name}", AsyncMock(side_effect=AssertionError(f"{name} called")))
    runner = _make_runner(_FakeGrader(), target)
    runner.suite.target.backend = "local"
    runner.suite.cleanup = cleanup
    runner.client = MagicMock()
    runner.client.agents.retrieve = AsyncMock()
    runner.client.agents.delete = AsyncMock()

    async def reward_composer(ctx):
        return RewardOutput(score=1.0)

    runner.reward_composer = reward_composer
    return runner


def _store(tmp_path: Path) -> str:
    store = tmp_path / "store"
    store.mkdir()
    return str(store)


@pytest.mark.asyncio
async def test_trace_comes_from_the_target(tmp_path, monkeypatch):
    store = _store(tmp_path)
    target = MagicMock()
    target.run = AsyncMock(
        return_value=TargetTrace(
            agent_id="agent-local-1",
            model_handle="llama.cpp/policy",
            agent_usage=USAGE,
            local_backend_dir=store,
            trajectory=[[MESSAGE]],
        )
    )
    runner = _local_runner(monkeypatch, target)

    result = await runner.run_sample(Sample(id=0, input="hi"), model_handle=None, return_token_data=True)

    assert result.error is None
    assert result.trajectory == [[MESSAGE]]
    assert result.token_data is None
    assert result.local_backend_dir == store
    runner.client.agents.retrieve.assert_not_awaited()
    assert Path(store).is_dir()


@pytest.mark.asyncio
async def test_failed_target_keeps_its_partial_trajectory(tmp_path, monkeypatch):
    store = _store(tmp_path)
    target = MagicMock()
    target.run = AsyncMock(
        side_effect=TargetError("boom", agent_id="agent-local-1", trajectory=[[MESSAGE]], local_backend_dir=store)
    )
    runner = _local_runner(monkeypatch, target)

    result = await runner.run_sample(Sample(id=0, input="hi"), model_handle=None, return_token_data=True)

    assert result.error is not None and result.error.message == "boom"
    assert result.trajectory == [[MESSAGE]]
    assert result.local_backend_dir == store


@pytest.mark.asyncio
async def test_cleanup_removes_the_local_store(tmp_path, monkeypatch):
    store = _store(tmp_path)
    target = MagicMock()
    target.run = AsyncMock(
        return_value=TargetTrace(
            agent_id="agent-local-1", model_handle="m", local_backend_dir=store, trajectory=[[MESSAGE]]
        )
    )
    runner = _local_runner(monkeypatch, target, cleanup=True)

    await runner.run_sample(Sample(id=0, input="hi"), model_handle=None)

    assert not Path(store).exists()
    runner.client.agents.delete.assert_not_awaited()
